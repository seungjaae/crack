"""GUI 의 순수 로직 (Qt 위젯 없이 검증 가능한 부분)."""

import pytest

pytest.importorskip("PySide6")

from crack.config import ColorSpec           # noqa: E402
from crack.gui import (                      # noqa: E402
    ranges_to_sliders,
    replace_ranges_in_toml,
    sliders_to_ranges,
)


def test_single_range_roundtrip():
    r = ((20, 80, 80, 35, 255, 255),)
    h_lo, h_hi, s, v = ranges_to_sliders(r)
    assert (h_lo, h_hi, s, v) == (20, 35, 80, 80)
    assert sliders_to_ranges(h_lo, h_hi, s, v) == r


def test_wraparound_range_roundtrip():
    """빨강은 H=0 을 감싸므로 구간이 2개다."""
    r = ((0, 80, 60, 10, 255, 255), (168, 80, 60, 179, 255, 255))
    h_lo, h_hi, s, v = ranges_to_sliders(r)
    assert (h_lo, h_hi) == (168, 10)      # 시작 > 끝 == 감싸는 구간
    assert sliders_to_ranges(h_lo, h_hi, s, v) == r


def test_slider_change_produces_valid_ranges():
    out = sliders_to_ranges(160, 15, 100, 70)
    assert out == ((0, 100, 70, 15, 255, 255), (160, 100, 70, 179, 255, 255))


def test_toml_rewrite_only_touches_target_color():
    text = (
        "[colors.red]\n"
        "hsv_ranges = [\n    [0, 80, 60, 10, 255, 255],\n]\n"
        "\n"
        "[colors.yellow]\n"
        "hsv_ranges = [\n    [20, 80, 80, 35, 255, 255],\n]\n"
    )
    spec = ColorSpec("red", ((0, 120, 90, 12, 255, 255),))
    out = replace_ranges_in_toml(text, "red", spec)

    import tomllib

    parsed = tomllib.loads(out)
    assert parsed["colors"]["red"]["hsv_ranges"] == [[0, 120, 90, 12, 255, 255]]
    assert parsed["colors"]["yellow"]["hsv_ranges"] == [[20, 80, 80, 35, 255, 255]]


def test_toml_rewrite_survives_comments():
    text = (
        "[colors.cyan]\n"
        "# 현장 락카 색에 맞춰 조정\n"
        "hsv_ranges = [\n    [85, 70, 60, 105, 255, 255],\n]\n"
    )
    spec = ColorSpec("cyan", ((90, 90, 70, 100, 255, 255),))
    out = replace_ranges_in_toml(text, "cyan", spec)

    import tomllib

    assert tomllib.loads(out)["colors"]["cyan"]["hsv_ranges"] == [
        [90, 90, 70, 100, 255, 255]
    ]
    assert "현장 락카 색에 맞춰 조정" in out


# ------------------------------------------------------- 선 클릭 판정
import numpy as np  # noqa: E402

from crack.gui import _point_to_polyline_px  # noqa: E402


def test_distance_to_a_segment_endpoint():
    poly = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert _point_to_polyline_px(np.array([0.0, 3.0]), poly) == pytest.approx(3.0)


def test_distance_projects_onto_the_segment():
    """선분 중간에 수직으로 떨어지는 거리를 재야 한다."""
    poly = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert _point_to_polyline_px(np.array([5.0, 4.0]), poly) == pytest.approx(4.0)


def test_distance_is_clamped_past_the_end():
    """선분을 벗어난 지점은 끝점까지의 거리로 잰다 (무한 직선이 아니다)."""
    poly = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert _point_to_polyline_px(np.array([13.0, 4.0]), poly) == pytest.approx(5.0)


def test_distance_uses_the_nearest_of_many_vertices():
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert _point_to_polyline_px(np.array([12.0, 5.0]), poly) == pytest.approx(2.0)


def test_degenerate_single_point_polyline():
    poly = np.array([[4.0, 3.0]])
    assert _point_to_polyline_px(np.array([0.0, 0.0]), poly) == pytest.approx(5.0)


def test_zero_length_segment_does_not_divide_by_zero():
    poly = np.array([[2.0, 2.0], [2.0, 2.0]])
    d = _point_to_polyline_px(np.array([2.0, 5.0]), poly)
    assert np.isfinite(d) and d == pytest.approx(3.0)


# ------------------------------------------- 실제 마우스 이벤트로 좌표 검증
@pytest.fixture(scope="session")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _label_with_pixmap(w: int, h: int, label_w: int, label_h: int):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    from crack.gui import ClickableLabel

    lbl = ClickableLabel(alignment=Qt.AlignCenter)
    pm = QPixmap(w, h)
    pm.fill()
    lbl.setPixmap(pm)
    lbl.resize(label_w, label_h)
    got: list[tuple[int, int]] = []
    lbl.clicked.connect(lambda x, y: got.append((x, y)))
    return lbl, got


def test_click_maps_through_centering_offset(qapp):
    """라벨이 이미지보다 크면 이미지가 가운데 정렬된다.

    그 여백을 빼지 않으면 클릭 좌표가 통째로 어긋나 선이 안 잡힌다.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    lbl, got = _label_with_pixmap(100, 80, 300, 200)
    # 이미지 좌상단은 라벨 기준 ((300-100)/2, (200-80)/2) = (100, 60)
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(110, 80))
    assert got == [(10, 20)]


def test_click_is_exact_when_label_matches_image(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    lbl, got = _label_with_pixmap(100, 80, 100, 80)
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(42, 17))
    assert got == [(42, 17)]


def test_click_outside_the_image_is_ignored(qapp):
    """여백을 눌렀을 때 엉뚱한 좌표를 흘리면 안 된다."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    lbl, got = _label_with_pixmap(100, 80, 300, 200)
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(5, 5))       # 좌측 여백
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(295, 195))   # 우측 여백
    assert got == []


def test_click_without_pixmap_does_nothing(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from crack.gui import ClickableLabel

    lbl = ClickableLabel()
    lbl.resize(100, 100)
    got = []
    lbl.clicked.connect(lambda x, y: got.append((x, y)))
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(50, 50))
    assert got == []


def test_click_maps_when_image_is_taller_than_the_label(qapp):
    """이미지가 라벨보다 크면 위아래가 잘린 채 가운데 정렬된다.

    잘려 나간 높이의 절반은 '음수 여백'이다. 이것을 0 으로 깎아 버리면
    클릭이 잘린 높이의 절반만큼 통째로 어긋난다 (실제로 겪은 버그).
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    lbl, got = _label_with_pixmap(100, 800, 300, 200)
    # 가로 여백 (300-100)/2 = +100, 세로 여백 (200-800)/2 = -300
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(110, 5))
    assert got == [(10, 305)]


def test_click_maps_when_image_is_wider_than_the_label(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    lbl, got = _label_with_pixmap(800, 100, 200, 300)
    # 가로 여백 (200-800)/2 = -300, 세로 여백 (300-100)/2 = +100
    QTest.mouseClick(lbl, Qt.LeftButton, pos=QPoint(5, 110))
    assert got == [(305, 10)]


# ------------------------------------------------- 놓친 선 직접 그리기
@pytest.fixture
def window(qapp, synthetic_deck):
    """추출은 돌리지 않고 빈 결과만 얹은 창. 그리기 동작만 본다."""
    import numpy as np

    from crack import config as config_mod
    from crack.gui import MainWindow
    from crack.pipeline import Result

    cfg = config_mod.load()
    w = MainWindow(cfg, None)
    w.resize(900, 700)
    w.show()
    w.load_image(str(synthetic_deck))
    w.result = Result(raster=w.raster, segments=[], label=np.zeros((1, 1), np.uint8))
    w._extract_done(w.result)
    return w


def test_drawn_line_becomes_a_manual_segment(window):
    w = window
    w.draw_grade.setCurrentIndex(w.draw_grade.findData("W2"))
    w.btn_draw.setChecked(True)
    assert w.drawing

    w._add_vertex(100, 100)
    w._add_vertex(200, 100)
    w.finish_drawing()

    assert len(w.result.segments) == 1
    s = w.result.segments[0]
    assert s.source == "manual"
    assert s.grade_id == "W2" and s.color == "yellow"

    # 미리보기 100px -> 원본 250px (축소배율 0.4) -> 125mm (GSD 0.5mm)
    assert s.length_mm == pytest.approx(125.0, rel=0.02)


def test_drawing_needs_at_least_two_points(window):
    w = window
    w.btn_draw.setChecked(True)
    w._add_vertex(50, 50)
    assert not w.btn_finish.isEnabled()
    w.finish_drawing()
    assert w.result.segments == []


def test_backspace_removes_the_last_vertex(window):
    w = window
    w.btn_draw.setChecked(True)
    for p in [(10, 10), (20, 20), (30, 30)]:
        w._add_vertex(*p)
    w.undo_vertex()
    assert len(w.draw_pts) == 2
    w.cancel_drawing()
    assert w.draw_pts == []


def test_drawn_segment_can_be_deleted_like_any_other(window):
    w = window
    w.btn_draw.setChecked(True)
    w._add_vertex(100, 100)
    w._add_vertex(300, 100)
    w.finish_drawing()
    w.btn_draw.setChecked(False)          # 그리기 모드를 꺼야 선택이 된다

    sid = w.result.segments[0].id
    w.selected = sid
    w.delete_selected()
    assert w.active_segments() == []
    w.undo_delete()
    assert len(w.active_segments()) == 1


def test_manual_segments_count_toward_the_stats(window):
    from crack import stats as stats_mod

    w = window
    w.btn_draw.setChecked(True)
    w._add_vertex(100, 100)
    w._add_vertex(300, 100)
    w.finish_drawing()

    summary = stats_mod.analyze(w.active_segments(), w.current_config())
    assert summary.count == 1
    assert summary.total_mm > 0
    assert "직접 그려 넣은 선 1개" in w.stats.toPlainText()


def test_source_defaults_to_auto():
    """자동 추출 세그먼트는 따로 표시하지 않아도 auto 여야 한다."""
    import numpy as np

    from crack.model import CrackSegment

    s = CrackSegment(
        id=1, grade_id="W1", grade_label="x", color="red", layer="L", dxf_color=1,
        length_mm=10.0, paint_width_mm=1.0, points_world=np.zeros((2, 2)),
    )
    assert s.source == "auto"
