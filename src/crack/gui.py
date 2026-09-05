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
from PySide6.QtGui import QAction, QImage, QPixmap
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import config as config_mod
from . import export, pipeline, segment
from .config import ColorSpec, Config
from .raster import Raster, open_raster, read_overview

PREVIEW_MAX_PX = 1600

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
        self.canvas = QLabel(alignment=Qt.AlignCenter)
        self.canvas.setMinimumSize(640, 480)
        self.canvas.setStyleSheet("background:#1b1b1b;")
        self.canvas.setText("정사영상을 열어 주세요")
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

        btn_save = QPushButton("현재 임계값을 설정 파일에 저장")
        btn_save.clicked.connect(self.save_config)
        lay.addWidget(btn_save)

        self.stats = QLabel("—")
        self.stats.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.stats.setWordWrap(True)
        lay.addWidget(self.stats)
        lay.addStretch(1)

        dock = QDockWidget("임계값 조정", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ------------------------------------------------------------- 상태
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
        )

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
        self.act_run.setEnabled(True)
        self.act_export.setEnabled(True)
        self.view_mode.setCurrentIndex(2)
        self.stats.setText(export.summarize(result.segments, self.current_config()))
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
        try:
            paths = [
                export.write_dxf(
                    self.result.segments,
                    cfg,
                    self.result.raster,
                    Path(out_dir) / f"{prefix}.dxf",
                ),
                export.write_tsv(
                    self.result.segments, cfg, Path(out_dir) / f"{prefix}.tsv"
                ),
            ]
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
        self.canvas.setPixmap(to_pixmap(img))
        self.canvas.resize(self.canvas.pixmap().size())

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
        inv = ~self.raster.transform
        for s in self.result.segments:
            pts = np.array([inv * (float(x), float(y)) for x, y in s.points_world])
            pts = np.round(pts * self.preview_scale).astype(np.int32)
            cv2.polylines(
                out,
                [pts],
                False,
                OVERLAY_RGB.get(s.color, (255, 255, 255)),
                2,
                cv2.LINE_AA,
            )
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
