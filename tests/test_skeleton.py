import cv2
import numpy as np

from crack.skeleton import _rdp, polyline_length, trace_segments


def test_rdp_drops_collinear_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    out = _rdp(pts, 0.5)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [0.0, 0.0])
    np.testing.assert_allclose(out[-1], [3.0, 0.0])


def test_rdp_keeps_a_real_corner():
    pts = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = _rdp(pts, 0.5)
    assert len(out) == 3


def test_polyline_length():
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert polyline_length(pts) == 5.0


def _draw(shape, polylines, thickness=20):
    img = np.zeros(shape, np.uint8)
    for pl in polylines:
        cv2.polylines(img, [np.array(pl, np.int32)], False, 255, thickness)
    return img


def test_straight_line_length_is_accurate():
    mask = _draw((400, 900), [[(100, 200), (800, 200)]])
    segs = trace_segments(mask, rdp_epsilon_px=1.5, prune_spur_px=10)
    assert len(segs) == 1
    # 끝단 캡 때문에 약간 짧게 나올 수 있으나 2% 이내여야 한다
    assert abs(segs[0]["length_px"] - 700) / 700 < 0.02
    assert abs(segs[0]["paint_width_px"] - 20) < 3


def test_branch_splits_into_three_segments():
    """Y자 균열은 분기점에서 3개로 쪼개져야 한다."""
    center = (500, 500)
    mask = _draw(
        (1000, 1000),
        [
            [(500, 150), center],
            [center, (200, 850)],
            [center, (850, 800)],
        ],
    )
    segs = trace_segments(mask, rdp_epsilon_px=1.5, prune_spur_px=10)
    assert len(segs) == 3

    lengths = sorted(s["length_px"] for s in segs)
    expected = sorted([350.0, np.hypot(300, 350), np.hypot(350, 300)])
    for got, want in zip(lengths, expected):
        assert abs(got - want) / want < 0.05


def test_empty_mask_returns_nothing():
    assert trace_segments(np.zeros((100, 100), np.uint8)) == []
