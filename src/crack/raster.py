"""GeoTIFF 입출력과 픽셀 <-> 실좌표 변환."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

# CRS 선형단위 -> mm 환산 계수
_UNIT_TO_MM = {
    "metre": 1000.0,
    "meter": 1000.0,
    "m": 1000.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "mm": 1.0,
    "centimetre": 10.0,
    "centimeter": 10.0,
    "cm": 10.0,
    "kilometre": 1_000_000.0,
    "kilometer": 1_000_000.0,
    "foot": 304.8,
    "ft": 304.8,
    "us survey foot": 304.80060960121924,
}


@dataclass
class Raster:
    """열려 있는 정사영상 한 장."""

    path: Path
    width: int
    height: int
    transform: object          # affine.Affine
    crs: object | None
    mm_per_unit: float         # 좌표계 1단위가 몇 mm 인가
    gsd_mm: float              # 픽셀 1개가 몇 mm 인가
    georeferenced: bool

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def px_to_world(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) [col,row] 픽셀좌표 -> (N,2) [x,y] 실좌표.

        픽셀 중심을 쓰기 위해 0.5 를 더한다.
        """
        pts = np.asarray(pts_px, dtype=np.float64)
        if pts.size == 0:
            return pts.reshape(0, 2)
        a, b, c, d, e, f = (
            self.transform.a,
            self.transform.b,
            self.transform.c,
            self.transform.d,
            self.transform.e,
            self.transform.f,
        )
        col = pts[:, 0] + 0.5
        row = pts[:, 1] + 0.5
        x = a * col + b * row + c
        y = d * col + e * row + f
        return np.column_stack([x, y])

    def world_len_to_mm(self, length_world: float) -> float:
        return length_world * self.mm_per_unit

    def mm_to_world_len(self, length_mm: float) -> float:
        return length_mm / self.mm_per_unit

    def read_rgb(self, window: Window | None = None) -> np.ndarray:
        """RGB 3밴드를 (H,W,3) uint8 로 읽는다."""
        with rasterio.open(self.path) as ds:
            n = min(3, ds.count)
            arr = ds.read(indexes=list(range(1, n + 1)), window=window)
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:            # 흑백이면 3채널로 복제
            arr = np.repeat(arr, 3, axis=2)
        return np.ascontiguousarray(arr)

    def tiles(self, tile_size: int, overlap: int) -> Iterator[tuple[Window, Window]]:
        """(읽을 창, 그 안에서 실제로 채택할 코어 영역) 쌍을 순회한다.

        코어 영역은 읽을 창 기준의 상대 좌표로 준다. 겹침 구간은
        모폴로지 연산이 타일 경계에서 깨지지 않게 하기 위한 여유분이다.
        """
        step = max(1, tile_size - 2 * overlap)
        for row0 in range(0, self.height, step):
            for col0 in range(0, self.width, step):
                r_start = max(0, row0 - overlap)
                c_start = max(0, col0 - overlap)
                r_end = min(self.height, row0 + step + overlap)
                c_end = min(self.width, col0 + step + overlap)
                read = Window(c_start, r_start, c_end - c_start, r_end - r_start)

                core_r0 = row0
                core_c0 = col0
                core_r1 = min(self.height, row0 + step)
                core_c1 = min(self.width, col0 + step)
                if core_r1 <= core_r0 or core_c1 <= core_c0:
                    continue
                core = Window(
                    core_c0 - c_start,
                    core_r0 - r_start,
                    core_c1 - core_c0,
                    core_r1 - core_r0,
                )
                yield read, core


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """uint8 이 아닌 래스터를 0~255 로 정규화."""
    out = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        out /= 257.0
    else:
        lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
        out = (out - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def open_raster(path: str | Path, fallback_gsd_mm: float = 0.5) -> Raster:
    """정사영상을 열고 좌표/해상도 메타데이터를 확정한다."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {p}")

    with rasterio.open(p) as ds:
        crs = ds.crs
        transform = ds.transform
        width, height = ds.width, ds.height

    georeferenced = crs is not None and not transform.is_identity

    if georeferenced:
        unit = (getattr(crs, "linear_units", "") or "").strip().lower()
        mm_per_unit = _UNIT_TO_MM.get(unit)
        if mm_per_unit is None:
            warnings.warn(
                f"좌표계 단위 '{unit}' 를 알 수 없어 미터로 가정합니다.", stacklevel=2
            )
            mm_per_unit = 1000.0
        gsd_mm = abs(transform.a) * mm_per_unit
    else:
        warnings.warn(
            f"좌표계가 없는 영상입니다. GSD {fallback_gsd_mm} mm/px 로 가정하고 "
            "원점을 좌측 하단으로 둔 밀리미터 좌표로 출력합니다.",
            stacklevel=2,
        )
        gsd_mm = fallback_gsd_mm
        mm_per_unit = 1.0
        # 좌표계가 없으면 밀리미터를 그대로 도면 좌표로 쓴다. 영상은 y 가 아래로
        # 증가하지만 도면은 위로 증가하므로 뒤집어서 원점을 좌측 하단에 둔다.
        transform = Affine(gsd_mm, 0.0, 0.0, 0.0, -gsd_mm, height * gsd_mm)

    return Raster(
        path=p,
        width=width,
        height=height,
        transform=transform,
        crs=crs,
        mm_per_unit=mm_per_unit,
        gsd_mm=gsd_mm,
        georeferenced=georeferenced,
    )


def read_overview(raster: Raster, max_px: int = 2000) -> tuple[np.ndarray, float]:
    """긴 변이 max_px 이하가 되도록 축소해서 읽는다.

    반환: (RGB (H,W,3) uint8, 축소배율). GeoTIFF 의 오버뷰를 활용하므로
    원본 전체를 읽지 않는다.
    """
    from rasterio.enums import Resampling

    scale = min(1.0, max_px / max(raster.width, raster.height))
    out_w = max(1, int(raster.width * scale))
    out_h = max(1, int(raster.height * scale))

    with rasterio.open(raster.path) as ds:
        n = min(3, ds.count)
        arr = ds.read(
            indexes=list(range(1, n + 1)),
            out_shape=(n, out_h, out_w),
            resampling=Resampling.bilinear,
        )
    img = np.transpose(arr, (1, 2, 0))
    if img.dtype != np.uint8:
        img = _to_uint8(np.transpose(img, (2, 0, 1))).transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img), scale
