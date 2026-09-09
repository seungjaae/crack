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
