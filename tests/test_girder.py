"""거더 검출 — 정답을 아는 합성 영상으로 검증한다."""

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from crack import config as config_mod
from crack import girder
from crack.raster import open_raster

W, H = 900, 1200
GSD_M = 0.0005


def make_girders(tmp_path, angle_deg=108.0, pitch=95, width=42, count=5,
                 occlude=None, gap_brightness=70):
    """평행한 밝은 띠 count 개를 그린 합성 정사영상.

    occlude 로 특정 거더의 일부를 어둡게 덮어 가려짐을 흉내 낼 수 있다.
    """
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(gap_brightness, 10, (H, W)), 0, 255).astype(np.uint8)
    img = np.repeat(img[:, :, None], 3, axis=2)

    u = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    n = np.array([-u[1], u[0]])
    centre = np.array([W / 2, H / 2])
    half_len = 900

    for i in range(count):
        off = (i - (count - 1) / 2) * pitch
        base = centre + off * n
        a = base - u * half_len
        b = base + u * half_len
        val = int(rng.integers(205, 235))
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 (val, val, val), width, cv2.LINE_AA)
        if occlude is not None and i == occlude:
            # 화면에 보이는 길이의 1/3 정도만 덮는다. 중앙값으로 판정하므로
            # 절반 넘게 가려진 거더는 원리상 찾을 수 없다.
            mid = base + u * (half_len * 0.1)
            cv2.line(img, tuple(np.round(mid - u * 200).astype(int)),
                     tuple(np.round(mid + u * 200).astype(int)),
                     (95, 95, 95), width + 6, cv2.LINE_AA)

    path = tmp_path / "girders.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
        crs=CRS.from_epsg(5186), transform=from_origin(200000.0, 550000.0, GSD_M, GSD_M),
    ) as ds:
        ds.write(np.transpose(img, (2, 0, 1)))
    return path


@pytest.fixture
def cfg():
    return config_mod.load()


def test_dominant_angle_is_found_to_sub_degree(tmp_path, cfg):
    """정수로 반올림하면 긴 거더의 끝단이 크게 밀린다."""
    p = make_girders(tmp_path, angle_deg=108.4)
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    got = girder.dominant_angle(gray, cfg)
    assert got is not None
    assert abs(got - 108.4) < 0.6


def test_finds_every_girder(tmp_path, cfg):
    p = make_girders(tmp_path, count=5)
    G = girder.detect(open_raster(p, cfg.fallback_gsd_mm), cfg)
    assert len(G) == 5


def test_width_and_spacing_are_right(tmp_path, cfg):
    p = make_girders(tmp_path, pitch=95, width=42, count=5)
    G = girder.detect(open_raster(p, cfg.fallback_gsd_mm), cfg)

    # 폭에는 설정상 여유(width_pad_px)가 양쪽으로 더해진다
    expect_mm = (42 + 2 * cfg.girder.width_pad_px) * 0.5
    for g in G:
        assert abs(g.width_mm - expect_mm) < 6.0

    centres = sorted(g.points_world[:, 0].mean() for g in G)
    gaps = np.diff(centres)
    assert gaps.std() < gaps.mean() * 0.1     # 등간격이어야 한다


def test_occluded_girder_keeps_the_common_length(tmp_path, cfg):
    """가려진 거더는 자체 측정이 짧게 나온다.

    한 경간의 거더는 길이가 거의 같으므로 합의 길이를 써야 한다.
    """
    p = make_girders(tmp_path, count=5, occlude=2)
    G = girder.detect(open_raster(p, cfg.fallback_gsd_mm), cfg)
    assert len(G) == 5
    lens = np.array([g.length_mm for g in G])
    assert lens.std() < lens.mean() * 0.1


def test_returns_nothing_on_a_blank_image(tmp_path, cfg):
    img = np.full((H, W, 3), 90, np.uint8)
    path = tmp_path / "flat.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
        crs=CRS.from_epsg(5186), transform=from_origin(200000.0, 550000.0, GSD_M, GSD_M),
    ) as ds:
        ds.write(np.transpose(img, (2, 0, 1)))
    assert girder.detect(open_raster(path, cfg.fallback_gsd_mm), cfg) == []


def test_quads_have_four_corners_in_world_coords(tmp_path, cfg):
    p = make_girders(tmp_path, count=4)
    G = girder.detect(open_raster(p, cfg.fallback_gsd_mm), cfg)
    assert G
    for g in G:
        assert g.points_world.shape == (4, 2)
        assert g.points_px.shape == (4, 2)
        assert g.wkt().startswith("POLYGON((")
        # EPSG:5186 좌표계 안에 들어와야 한다
        assert 199_000 < g.points_world[:, 0].mean() < 201_000
