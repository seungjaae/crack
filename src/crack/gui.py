"""임계값 튜닝용 데스크톱 GUI.

현장 조도나 락카 상태에 따라 색상 임계값은 반드시 조정이 필요하다.
축소본 위에서 슬라이더를 움직이며 즉시 결과를 확인하고,
확정되면 원본 해상도로 추출한 뒤 DXF/TSV 로 내보낸다.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import config as config_mod
from . import export, pipeline, segment, stats
from .config import SORT_KEYS, ColorSpec, Config
from .model import CrackSegment
from .raster import Raster, open_raster, read_overview
from .skeleton import polyline_length

PREVIEW_MAX_PX = 1600

# 선을 고를 때 허용하는 클릭 오차 (미리보기 픽셀).
# 균열선은 화면에서 2px 남짓으로 가늘기 때문에 넉넉히 잡아야 집힌다.
CLICK_RADIUS_PX = 22.0

OVERLAY_RGB = {
    "red": (255, 70, 60),
    "yellow": (255, 214, 40),
    "cyan": (60, 200, 255),
}


# --------------------------------------------------------------------------
# HSV 범위 <-> 슬라이더 값 변환
# 빨강처럼 H=0 을 감싸는 색은 구간이 2개다. 이를 (h_lo > h_hi) 한 쌍으로 다룬다.
# --------------------------------------------------------------------------
def ranges_to_sliders(ranges) -> tuple[int, int, int, int]:
    if len(ranges) == 1:
        h0, s0, v0, h1, _s1, _v1 = ranges[0]
        return h0, h1, s0, v0
    wrap_low = max(r[3] for r in ranges if r[0] == 0)
    wrap_high = min(r[0] for r in ranges if r[0] > 0)
    return wrap_high, wrap_low, ranges[0][1], ranges[0][2]


def sliders_to_ranges(h_lo: int, h_hi: int, s_min: int, v_min: int):
    if h_lo <= h_hi:
        return ((h_lo, s_min, v_min, h_hi, 255, 255),)
    return (
        (0, s_min, v_min, h_hi, 255, 255),
        (h_lo, s_min, v_min, 179, 255, 255),
    )


class ColorControls(QWidget):
    """색상 하나의 HSV 임계값 슬라이더 묶음."""

    changed = Signal()

    def __init__(self, name: str, spec: ColorSpec, parent=None):
        super().__init__(parent)
        self.name = name
        h_lo, h_hi, s_min, v_min = ranges_to_sliders(spec.hsv_ranges)

        form = QFormLayout(self)
        self.h_lo = self._slider(0, 179, h_lo)
        self.h_hi = self._slider(0, 179, h_hi)
        self.s_min = self._slider(0, 255, s_min)
        self.v_min = self._slider(0, 255, v_min)

        form.addRow("색상 시작 (H)", self._row(self.h_lo))
        form.addRow("색상 끝 (H)", self._row(self.h_hi))
        form.addRow("최소 채도 (S)", self._row(self.s_min))
        form.addRow("최소 명도 (V)", self._row(self.v_min))

        hint = QLabel(
            "채도(S)를 올리면 회색 콘크리트가 확실히 빠집니다.\n"
            "명도(V)를 내리면 그늘 속 락카도 잡힙니다.\n"
            "시작 > 끝 이면 H=0 을 감싸는 구간(빨강)이 됩니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-size: 11px;")
        form.addRow(hint)

    def _slider(self, lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(int(val))
        s.valueChanged.connect(self.changed)
        return s

    def _row(self, slider: QSlider) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        label = QLabel(str(slider.value()))
        label.setMinimumWidth(30)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v, lb=label: lb.setText(str(v)))
        lay.addWidget(slider)
        lay.addWidget(label)
        return box

    def to_spec(self) -> ColorSpec:
        return ColorSpec(
            name=self.name,
            hsv_ranges=sliders_to_ranges(
                self.h_lo.value(),
                self.h_hi.value(),
                self.s_min.value(),
                self.v_min.value(),
            ),
        )


class ClickableLabel(QLabel):
    """캔버스. 클릭 위치를 미리보기 픽셀 좌표로 바꿔서 알린다."""

    clicked = Signal(int, int)
    double_clicked = Signal()

    def _to_image_px(self, ev) -> tuple[int, int] | None:
        """위젯 좌표를 이미지 픽셀 좌표로. 이미지 바깥이면 None."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None

        # 이미지는 라벨 안에서 가운데 정렬된다. 그 여백을 빼야 좌표가 맞는다.
        # 이미지가 라벨보다 크면 여백은 음수가 된다 (위아래가 잘려 나간 만큼).
        # 여기서 0 으로 깎으면 잘린 높이의 절반만큼 클릭이 통째로 어긋난다.
        dpr = pm.devicePixelRatio() or 1.0
        pw, ph = pm.width() / dpr, pm.height() / dpr
        off_x = (self.width() - pw) / 2.0
        off_y = (self.height() - ph) / 2.0

        pos = ev.position()
        x, y = pos.x() - off_x, pos.y() - off_y
        if 0.0 <= x < pw and 0.0 <= y < ph:
            return int(x), int(y)
        return None

    def mousePressEvent(self, ev) -> None:  # noqa: N802 - Qt 규약
        hit = self._to_image_px(ev)
        if hit is not None:
            self.clicked.emit(*hit)

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802 - Qt 규약
        if self._to_image_px(ev) is not None:
            self.double_clicked.emit()


def _point_to_polyline_px(pt: np.ndarray, poly: np.ndarray) -> float:
    """점에서 폴리라인까지의 최단 거리 (픽셀)."""
    if len(poly) == 1:
        return float(np.hypot(*(pt - poly[0])))
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = (ab * ab).sum(axis=1)
    denom[denom == 0] = 1e-9
    t = np.clip(((pt - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.hypot(proj[:, 0] - pt[0], proj[:, 1] - pt[1]).min())


class ExtractWorker(QThread):
    """원본 해상도 추출은 UI 를 막지 않도록 별도 스레드에서."""

    progress = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, image_path: Path, cfg: Config):
        super().__init__()
        self.image_path = image_path
        self.cfg = cfg

    def run(self) -> None:
        try:
            res = pipeline.run(self.image_path, self.cfg, progress=self.progress.emit)
            self.finished_ok.emit(res)
        except Exception as e:  # noqa: BLE001 - 사용자에게 그대로 보여준다
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, config_path: Path | None):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path or config_mod.DEFAULT_CONFIG
        self.raster: Raster | None = None
        self.preview_rgb: np.ndarray | None = None
        self.preview_scale = 1.0
        self.result = None
        self.worker: ExtractWorker | None = None

        # 잘못 추출된 선 관리
        self.removed: set[int] = set()          # 지운 세그먼트 id
        self.undo_stack: list[int] = []         # 지운 순서 (되돌리기용)
        self.selected: int | None = None        # 현재 선택된 세그먼트 id
        self._preview_pts: dict[int, np.ndarray] = {}   # id -> 미리보기 좌표

        # 놓친 선 직접 그리기
        self.drawing = False
        self.draw_pts: list[tuple[float, float]] = []   # 미리보기 좌표

        self.setWindowTitle("Concrete to Code — 균열 추출 엔진")
        self.resize(1400, 900)

        # 슬라이더를 끌 때마다 다시 그리면 버벅이므로 잠깐 모았다가 한 번만 그린다.
        # 위젯 구성 중에 연결되므로 반드시 먼저 만들어 둔다.
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(120)
        self.debounce.timeout.connect(self._refresh_canvas)

        self._build_toolbar()
        self._build_canvas()
        self._build_dock()

        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence.Undo, self, self.undo_delete)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.finish_drawing)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.finish_drawing)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.cancel_drawing)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.undo_vertex)

        self.statusBar().showMessage("GeoTIFF 정사영상을 열어 주세요.")

    # ---------------------------------------------------------------- UI
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)

        act_open = QAction("열기", self)
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        tb.addSeparator()
        self.act_run = QAction("추출 실행", self)
        self.act_run.triggered.connect(self.run_extract)
        self.act_run.setEnabled(False)
        tb.addAction(self.act_run)

        self.act_export = QAction("DXF / TSV 내보내기", self)
        self.act_export.triggered.connect(self.export_results)
        self.act_export.setEnabled(False)
        tb.addAction(self.act_export)

        tb.addSeparator()
        tb.addWidget(QLabel("  보기: "))
        self.view_mode = QComboBox()
        self.view_mode.addItems(["원본", "색상 마스크", "추출 결과"])
        self.view_mode.currentIndexChanged.connect(self._refresh_canvas)
        tb.addWidget(self.view_mode)

    def _build_canvas(self) -> None:
        self.canvas = ClickableLabel(alignment=Qt.AlignCenter)
        self.canvas.setMinimumSize(640, 480)
        self.canvas.setStyleSheet("background:#1b1b1b;")
        self.canvas.setText("정사영상을 열어 주세요")
        self.canvas.clicked.connect(self._on_canvas_click)
        self.canvas.double_clicked.connect(self.finish_drawing)
        area = QScrollArea()
        area.setWidget(self.canvas)
        area.setWidgetResizable(True)
        self.setCentralWidget(area)

    def _build_dock(self) -> None:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        self.tabs = QTabWidget()
        self.controls: dict[str, ColorControls] = {}
        for grade in self.cfg.grades:
            c = ColorControls(grade.color, self.cfg.colors[grade.color])
            c.changed.connect(self.debounce.start)
            self.controls[grade.color] = c
            self.tabs.addTab(c, f"{grade.id} · {grade.color}")
        lay.addWidget(self.tabs)

        box = QGroupBox("노이즈 필터")
        form = QFormLayout(box)
        self.min_length = QSpinBox()
        self.min_length.setRange(0, 100000)
        self.min_length.setValue(int(self.cfg.denoise.min_length_mm))
        self.min_length.setSuffix(" mm")
        self.min_elong = QSpinBox()
        self.min_elong.setRange(1, 100)
        self.min_elong.setValue(int(self.cfg.denoise.min_elongation))
        form.addRow("최소 길이", self.min_length)
        form.addRow("최소 세장비", self.min_elong)
        lay.addWidget(box)

        self.chk_text = QCheckBox("DXF 에 길이 텍스트 병기")
        self.chk_text.setChecked(self.cfg.export.write_text)
        lay.addWidget(self.chk_text)

        sort_box = QGroupBox("속성값 분석")
        sform = QFormLayout(sort_box)
        self.sort_mode = QComboBox()
        for key, label in (
            ("grade_length", "색상순 · 긴 것부터"),
            ("grade", "색상순"),
            ("length_desc", "긴 것부터"),
            ("length", "짧은 것부터"),
            ("id", "추출 순서"),
        ):
            self.sort_mode.addItem(label, key)
        idx = self.sort_mode.findData(self.cfg.analysis.sort_by)
        self.sort_mode.setCurrentIndex(max(0, idx))
        self.sort_mode.currentIndexChanged.connect(self._refresh_stats)
        sform.addRow("정렬", self.sort_mode)

        self.bins_edit = QLineEdit(
            ", ".join(f"{b:g}" for b in self.cfg.analysis.length_bins_mm)
        )
        self.bins_edit.setPlaceholderText("예: 300, 600, 1000, 2000")
        self.bins_edit.editingFinished.connect(self._refresh_stats)
        sform.addRow("길이 구간(mm)", self.bins_edit)
        lay.addWidget(sort_box)

        # ---- 색상별 보기 ----
        vis_box = QGroupBox("표시할 색상")
        vlay = QHBoxLayout(vis_box)
        self.vis: dict[str, QCheckBox] = {}
        for grade in self.cfg.grades:
            cb = QCheckBox(f"{grade.id}·{grade.color}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_canvas)
            self.vis[grade.id] = cb
            vlay.addWidget(cb)
        lay.addWidget(vis_box)

        # ---- 잘못 추출된 선 제거 ----
        del_box = QGroupBox("선 편집")
        dlay = QVBoxLayout(del_box)
        self.sel_label = QLabel("선을 클릭해 선택하세요")
        self.sel_label.setStyleSheet("color:#777; font-size:11px;")
        self.sel_label.setWordWrap(True)
        dlay.addWidget(self.sel_label)

        btns = QHBoxLayout()
        self.btn_del = QPushButton("선택 삭제")
        self.btn_del.setEnabled(False)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_undo = QPushButton("되돌리기")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_delete)
        self.btn_restore = QPushButton("전체 복원")
        self.btn_restore.setEnabled(False)
        self.btn_restore.clicked.connect(self.restore_all)
        for b in (self.btn_del, self.btn_undo, self.btn_restore):
            btns.addWidget(b)
        dlay.addLayout(btns)

        hint = QLabel("단축키: Delete 삭제 · Ctrl+Z 되돌리기\n"
                      "삭제한 선은 통계와 내보내기에서 함께 빠집니다.")
        hint.setStyleSheet("color:#777; font-size:11px;")
        hint.setWordWrap(True)
        dlay.addWidget(hint)
        lay.addWidget(del_box)

        # ---- 놓친 선 직접 그려 넣기 ----
        # 마커 붓칠이 흐리거나 끊긴 구간은 자동 추출이 놓친다. 사람이 보완한다.
        add_box = QGroupBox("선 추가")
        alay = QVBoxLayout(add_box)

        top = QHBoxLayout()
        self.btn_draw = QPushButton("그리기 시작")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self._toggle_draw_mode)
        top.addWidget(self.btn_draw)
        top.addWidget(QLabel("색상:"))
        self.draw_grade = QComboBox()
        for grade in self.cfg.grades:
            self.draw_grade.addItem(f"{grade.id}·{grade.color}", grade.id)
        top.addWidget(self.draw_grade, 1)
        alay.addLayout(top)

        self.draw_label = QLabel("추출이 놓친 균열을 직접 그려 넣을 수 있습니다.")
        self.draw_label.setStyleSheet("color:#777; font-size:11px;")
        self.draw_label.setWordWrap(True)
        alay.addWidget(self.draw_label)

        dbtns = QHBoxLayout()
        self.btn_finish = QPushButton("선 완료")
        self.btn_finish.setEnabled(False)
        self.btn_finish.clicked.connect(self.finish_drawing)
        self.btn_cancel = QPushButton("그리기 취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_drawing)
        dbtns.addWidget(self.btn_finish)
        dbtns.addWidget(self.btn_cancel)
        alay.addLayout(dbtns)

        dhint = QLabel("클릭으로 점을 찍고, 더블클릭 또는 Enter 로 마칩니다.\n"
                       "Backspace 마지막 점 취소 · Esc 그리기 취소")
        dhint.setStyleSheet("color:#777; font-size:11px;")
        dhint.setWordWrap(True)
        alay.addWidget(dhint)
        lay.addWidget(add_box)

        btn_save = QPushButton("현재 임계값을 설정 파일에 저장")
        btn_save.clicked.connect(self.save_config)
        lay.addWidget(btn_save)

        self.stats = QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.stats.setPlaceholderText("추출을 실행하면 속성값 분석 결과가 표시됩니다.")
        self.stats.setStyleSheet(
            "font-family: 'D2Coding', Consolas, monospace; font-size: 11px;"
        )
        self.stats.setMinimumHeight(260)
        lay.addWidget(self.stats, 1)

        dock = QDockWidget("임계값 조정", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ------------------------------------------------------------- 상태
    def _parse_bins(self) -> tuple[float, ...]:
        """길이 구간 입력을 읽는다. 잘못 적었으면 기존 값을 그대로 쓴다."""
        try:
            vals = [
                float(t) for t in self.bins_edit.text().replace(" ", "").split(",") if t
            ]
        except ValueError:
            return self.cfg.analysis.length_bins_mm
        vals = sorted(v for v in vals if v > 0)
        return tuple(vals) if vals else self.cfg.analysis.length_bins_mm

    def current_config(self) -> Config:
        colors = dict(self.cfg.colors)
        for name, ctrl in self.controls.items():
            colors[name] = ctrl.to_spec()
        return replace(
            self.cfg,
            colors=colors,
            denoise=replace(
                self.cfg.denoise,
                min_length_mm=float(self.min_length.value()),
                min_elongation=float(self.min_elong.value()),
            ),
            export=replace(self.cfg.export, write_text=self.chk_text.isChecked()),
            analysis=replace(
                self.cfg.analysis,
                sort_by=self.sort_mode.currentData() or self.cfg.analysis.sort_by,
                length_bins_mm=self._parse_bins(),
            ),
        )

    def active_segments(self) -> list:
        """삭제되지 않은 세그먼트만. 통계와 내보내기는 항상 이것을 쓴다."""
        if self.result is None:
            return []
        return [s for s in self.result.segments if s.id not in self.removed]

    def _refresh_stats(self) -> None:
        """정렬 기준이나 길이 구간이 바뀌면 재추출 없이 분석만 다시 한다."""
        if self.result is None:
            return
        cfg = self.current_config()
        segs = stats.sort_segments(self.active_segments(), cfg)
        text = stats.format_report(stats.analyze(segs, cfg), cfg)
        notes = []
        if self.removed:
            notes.append(f"사용자가 삭제한 선 {len(self.removed)}개는 제외된 값입니다.")
        n_manual = sum(1 for s in segs if s.source == "manual")
        if n_manual:
            notes.append(f"직접 그려 넣은 선 {n_manual}개가 포함되어 있습니다.")
        if notes:
            text += "\n\n" + "\n".join(f"  * {n}" for n in notes)
        self.stats.setPlainText(text)

    # -------------------------------------------------------- 선 편집
    def _build_preview_pts(self) -> None:
        """세그먼트를 미리보기 픽셀 좌표로 미리 변환해 둔다 (클릭 판정용)."""
        self._preview_pts = {}
        if self.result is None or self.raster is None:
            return
        inv = ~self.raster.transform
        for s in self.result.segments:
            pts = np.array([inv @ (float(x), float(y)) for x, y in s.points_world])
            self._preview_pts[s.id] = pts * self.preview_scale

    def _on_canvas_click(self, x: int, y: int) -> None:
        """그리기 중이면 점을 찍고, 아니면 가장 가까운 선을 고른다."""
        if self.drawing:
            self._add_vertex(x, y)
            return
        if self.result is None or self.view_mode.currentIndex() != 2:
            return
        pt = np.array([float(x), float(y)])
        visible = {g.id for g in self.cfg.grades if self.vis[g.id].isChecked()}

        best, best_d = None, 1e18
        for s in self.active_segments():
            if s.grade_id not in visible:
                continue
            poly = self._preview_pts.get(s.id)
            if poly is None or len(poly) == 0:
                continue
            d = _point_to_polyline_px(pt, poly)
            if d < best_d:
                best, best_d = s, d

        # 너무 멀리 찍으면 선택 해제
        self.selected = best.id if (best is not None and best_d <= CLICK_RADIUS_PX) else None
        if self.selected is not None and best is not None:
            mark = " · 직접 그림" if best.source == "manual" else ""
            self.sel_label.setText(
                f"선택: #{best.id}  {best.grade_id}·{best.color}  "
                f"길이 {best.length_mm:,.1f} mm{mark}"
            )
        elif best is not None:
            self.sel_label.setText(
                f"가까운 선이 없습니다 (가장 가까운 선까지 {best_d:.0f}px). "
                "선 위를 눌러 주세요."
            )
        else:
            self.sel_label.setText("선을 클릭해 선택하세요")
        self._update_edit_buttons()
        self._refresh_canvas()

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self.removed.add(self.selected)
        self.undo_stack.append(self.selected)
        self.selected = None
        self.sel_label.setText(f"삭제됨. 총 {len(self.removed)}개 제외 중")
        self._after_edit()

    def undo_delete(self) -> None:
        if not self.undo_stack:
            return
        sid = self.undo_stack.pop()
        self.removed.discard(sid)
        self.selected = sid
        self.sel_label.setText(f"#{sid} 복원됨. 총 {len(self.removed)}개 제외 중")
        self._after_edit()

    def restore_all(self) -> None:
        self.removed.clear()
        self.undo_stack.clear()
        self.sel_label.setText("전체 복원됨")
        self._after_edit()

    def _after_edit(self) -> None:
        self._update_edit_buttons()
        self._refresh_stats()
        self._refresh_canvas()
        n = len(self.active_segments())
        self.statusBar().showMessage(
            f"세그먼트 {n}개 (삭제 {len(self.removed)}개 제외)"
        )

    def _update_edit_buttons(self) -> None:
        self.btn_del.setEnabled(self.selected is not None)
        self.btn_undo.setEnabled(bool(self.undo_stack))
        self.btn_restore.setEnabled(bool(self.removed))

    # -------------------------------------------------------- 선 추가
    def _toggle_draw_mode(self, on: bool) -> None:
        self.drawing = on
        self.draw_pts.clear()
        self.btn_draw.setText("그리기 종료" if on else "그리기 시작")
        self.btn_finish.setEnabled(False)
        self.btn_cancel.setEnabled(on)
        if on:
            self.selected = None
            self.view_mode.setCurrentIndex(2)     # 결과 화면 위에서만 그린다
            self.draw_label.setText("캔버스를 클릭해 점을 찍으세요.")
        else:
            self.draw_label.setText("추출이 놓친 균열을 직접 그려 넣을 수 있습니다.")
        self._update_edit_buttons()
        self._refresh_canvas()

    def _draw_length_mm(self) -> float:
        if self.raster is None or len(self.draw_pts) < 2:
            return 0.0
        img_px = np.array(self.draw_pts) / self.preview_scale
        return float(polyline_length(img_px) * self.raster.gsd_mm)

    def _draw_status(self) -> str:
        if not self.draw_pts:
            return "캔버스를 클릭해 점을 찍으세요."
        return f"{len(self.draw_pts)}점 · 길이 {self._draw_length_mm():,.0f} mm"

    def _add_vertex(self, x: int, y: int) -> None:
        self.draw_pts.append((float(x), float(y)))
        self.btn_finish.setEnabled(len(self.draw_pts) >= 2)
        self.draw_label.setText(self._draw_status())
        self._refresh_canvas()

    def undo_vertex(self) -> None:
        if not (self.drawing and self.draw_pts):
            return
        self.draw_pts.pop()
        self.btn_finish.setEnabled(len(self.draw_pts) >= 2)
        self.draw_label.setText(self._draw_status())
        self._refresh_canvas()

    def finish_drawing(self) -> None:
        """찍은 점들로 세그먼트를 하나 만들어 결과에 넣는다."""
        if not self.drawing or len(self.draw_pts) < 2 or self.result is None:
            return
        gid = self.draw_grade.currentData()
        grade = next(g for g in self.cfg.grades if g.id == gid)

        img_px = np.array(self.draw_pts) / self.preview_scale
        new_id = max((s.id for s in self.result.segments), default=0) + 1
        seg = CrackSegment(
            id=new_id,
            grade_id=grade.id,
            grade_label=grade.label,
            color=grade.color,
            layer=grade.layer,
            dxf_color=grade.dxf_color,
            length_mm=self._draw_length_mm(),
            paint_width_mm=0.0,        # 사람이 그린 선이라 칠 폭은 알 수 없다
            points_world=self.raster.px_to_world(img_px),
            source="manual",
        )
        self.result.segments.append(seg)
        self._preview_pts[seg.id] = np.array(self.draw_pts)

        self.draw_pts.clear()
        self.btn_finish.setEnabled(False)
        self.draw_label.setText(
            f"#{seg.id} 추가됨 · {grade.id}·{grade.color} · "
            f"{seg.length_mm:,.0f} mm. 이어서 그릴 수 있습니다."
        )
        self._refresh_stats()
        self._refresh_canvas()
        self.statusBar().showMessage(
            f"세그먼트 {len(self.active_segments())}개 "
            f"(직접 추가 {sum(1 for s in self.active_segments() if s.source == 'manual')}개)"
        )

    def cancel_drawing(self) -> None:
        self.draw_pts.clear()
        self.btn_finish.setEnabled(False)
        self.draw_label.setText(self._draw_status() if self.drawing
                                else "추출이 놓친 균열을 직접 그려 넣을 수 있습니다.")
        self._refresh_canvas()

    # ------------------------------------------------------------ 액션
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "정사영상 열기", "", "GeoTIFF (*.tif *.tiff);;모든 파일 (*)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str | Path) -> None:
        path = str(path)
        try:
            self.raster = open_raster(path, self.cfg.fallback_gsd_mm)
            self.preview_rgb, self.preview_scale = read_overview(
                self.raster, PREVIEW_MAX_PX
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "열기 실패", str(e))
            return

        self.result = None
        self.act_run.setEnabled(True)
        self.act_export.setEnabled(False)
        crs = self.raster.crs if self.raster.georeferenced else "좌표계 없음"
        self.statusBar().showMessage(
            f"{Path(path).name}  |  {self.raster.width}x{self.raster.height} px "
            f"({self.raster.megapixels:.1f} MP)  |  "
            f"GSD {self.raster.gsd_mm:.3f} mm/px  |  {crs}"
        )
        self._refresh_canvas()

    def run_extract(self) -> None:
        if self.raster is None or self.worker is not None:
            return
        self.act_run.setEnabled(False)
        self.worker = ExtractWorker(self.raster.path, self.current_config())
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished_ok.connect(self._extract_done)
        self.worker.failed.connect(self._extract_failed)
        self.worker.start()

    def _extract_done(self, result) -> None:
        self.result = result
        self.worker = None
        # 새로 추출했으므로 이전 편집 이력은 의미가 없다
        self.removed.clear()
        self.undo_stack.clear()
        self.selected = None
        self.sel_label.setText("선을 클릭해 선택하세요")
        self._update_edit_buttons()
        self._build_preview_pts()

        self.act_run.setEnabled(True)
        self.act_export.setEnabled(True)
        self.view_mode.setCurrentIndex(2)
        self._refresh_stats()
        self.statusBar().showMessage(
            f"추출 완료 — 세그먼트 {len(result.segments)}개, {result.elapsed_s:.1f}초"
        )
        self._refresh_canvas()

    def _extract_failed(self, msg: str) -> None:
        self.worker = None
        self.act_run.setEnabled(True)
        QMessageBox.critical(self, "추출 실패", msg)

    def export_results(self) -> None:
        if self.result is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", "")
        if not out_dir:
            return
        cfg = self.current_config()
        prefix = self.raster.path.stem
        segs = stats.sort_segments(self.active_segments(), cfg)
        summary = stats.analyze(segs, cfg)
        try:
            paths = [
                export.write_dxf(
                    segs, cfg, self.result.raster, Path(out_dir) / f"{prefix}.dxf"
                ),
                export.write_tsv(segs, cfg, Path(out_dir) / f"{prefix}.tsv"),
                export.write_stats_tsv(
                    summary, cfg, Path(out_dir) / f"{prefix}_stats.tsv"
                ),
                export.write_preview(
                    segs, cfg, self.result.raster,
                    Path(out_dir) / f"{prefix}_preview.png",
                ),
            ]
            paths += export.write_previews_by_color(
                segs, cfg, self.result.raster, Path(out_dir), prefix
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "내보내기 실패", str(e))
            return
        QMessageBox.information(
            self, "완료", "저장했습니다.\n\n" + "\n".join(str(p) for p in paths)
        )

    def save_config(self) -> None:
        cfg = self.current_config()
        try:
            text = self.config_path.read_text(encoding="utf-8")
            for name, spec in cfg.colors.items():
                text = replace_ranges_in_toml(text, name, spec)
            self.config_path.write_text(text, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", str(e))
            return
        QMessageBox.information(self, "저장됨", f"{self.config_path} 에 저장했습니다.")

    # ------------------------------------------------------------ 렌더
    def _refresh_canvas(self) -> None:
        if self.preview_rgb is None:
            return
        mode = self.view_mode.currentIndex()
        if mode == 0:
            img = self.preview_rgb
        elif mode == 1:
            img = self._mask_overlay()
        else:
            img = self._result_overlay()
        pm = to_pixmap(img)
        self.canvas.setPixmap(pm)
        # 라벨이 이미지보다 작으면 위아래가 잘린 채 스크롤도 안 된다.
        # 최소 크기를 이미지에 맞춰 두면 스크롤 영역이 스크롤바를 내준다.
        self.canvas.setMinimumSize(pm.size())

    def _mask_overlay(self) -> np.ndarray:
        """축소본 위에서 현재 임계값의 색상 검출 결과를 즉시 보여준다."""
        cfg = self.current_config()
        out = (self.preview_rgb * 0.35).astype(np.uint8)
        for grade in cfg.grades:
            mask = segment.color_mask(
                self.preview_rgb, cfg.colors[grade.color].hsv_ranges, cfg
            )
            out[mask > 0] = OVERLAY_RGB.get(grade.color, (255, 255, 255))
        return out

    def _result_overlay(self) -> np.ndarray:
        if self.result is None or self.raster is None:
            return self.preview_rgb
        out = (self.preview_rgb * 0.4).astype(np.uint8)
        visible = {g.id for g in self.cfg.grades if self.vis[g.id].isChecked()}

        def draw(sid, color, thickness):
            poly = self._preview_pts.get(sid)
            if poly is None or len(poly) < 2:
                return
            cv2.polylines(out, [np.round(poly).astype(np.int32)], False,
                          color, thickness, cv2.LINE_AA)

        # 삭제한 선은 지운 자리를 알 수 있게 흐리게 남겨 둔다
        for s in self.result.segments:
            if s.id in self.removed and s.grade_id in visible:
                draw(s.id, (110, 110, 110), 1)

        for s in self.active_segments():
            if s.grade_id in visible:
                draw(s.id, OVERLAY_RGB.get(s.color, (255, 255, 255)), 2)

        if self.selected is not None and self.selected not in self.removed:
            draw(self.selected, (255, 255, 255), 4)

        # 그리는 중인 선: 찍은 점과 이어진 구간을 바로 보여 준다
        if self.drawing and self.draw_pts:
            gid = self.draw_grade.currentData()
            grade = next((g for g in self.cfg.grades if g.id == gid), None)
            col = OVERLAY_RGB.get(grade.color if grade else "", (255, 255, 255))
            pts = np.round(np.array(self.draw_pts)).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(out, [pts], False, col, 2, cv2.LINE_AA)
            for i, (px, py) in enumerate(pts):
                cv2.circle(out, (int(px), int(py)), 4, (255, 255, 255), -1)
                cv2.circle(out, (int(px), int(py)), 4, col, 1)
                if i == 0:
                    cv2.circle(out, (int(px), int(py)), 7, (255, 255, 255), 1)
        return out


def replace_ranges_in_toml(text: str, color: str, spec: ColorSpec) -> str:
    """TOML 안의 [colors.<name>] hsv_ranges 블록만 교체한다."""
    body = ",\n".join(
        "    [" + ", ".join(f"{v:>3}" for v in r) + "]" for r in spec.hsv_ranges
    )
    new_block = f"hsv_ranges = [\n{body},\n]"
    pattern = re.compile(
        r"(\[colors\." + re.escape(color) + r"\]\s*\n(?:[ \t]*#[^\n]*\n)*[ \t]*)"
        r"hsv_ranges\s*=\s*\[.*?\][ \t]*\n",
        re.DOTALL,
    )
    return pattern.sub(lambda m: m.group(1) + new_block + "\n", text, count=1)


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    cfg_path = None
    if "--config" in argv:
        i = argv.index("--config")
        cfg_path = Path(argv[i + 1])
        del argv[i : i + 2]

    image = next((a for a in argv[1:] if not a.startswith("-")), None)

    cfg = config_mod.load(cfg_path)
    app = QApplication(argv[:1])
    win = MainWindow(cfg, cfg_path)
    win.show()
    if image:
        win.load_image(image)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
