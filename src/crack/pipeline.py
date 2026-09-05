"""전체 처리 흐름 조립.

정사영상 -> 색상 라벨맵 -> 등급별 마스크 -> 세그먼트 -> 실좌표 균열
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import segment as seg_mod
from . import skeleton as skel_mod
from .config import Config
from .model import CrackSegment
from .raster import Raster, open_raster

Progress = Callable[[str], None]


@dataclass
class Result:
    raster: Raster
    segments: list[CrackSegment]
    label: np.ndarray
    elapsed_s: float = 0.0
    dropped: dict[str, int] = field(default_factory=dict)


def build_label_map(
    raster: Raster,
    cfg: Config,
    tile_size: int = 4096,
    progress: Progress = lambda _m: None,
) -> np.ndarray:
    """영상 전체에 대한 등급 라벨맵(uint8)을 만든다.

    RGB 원본을 통째로 올리지 않고 타일 단위로 읽어서 메모리를 아낀다.
    라벨맵은 화소당 1바이트라 RGB 대비 1/3 이다.
    """
    overlap = max(cfg.denoise.morph_open_px, cfg.denoise.morph_close_px) * 2 + 4

    if raster.width <= tile_size and raster.height <= tile_size:
        progress(f"  영상 일괄 처리 ({raster.width}x{raster.height})")
        return seg_mod.build_label(raster.read_rgb(), cfg)

    label = np.zeros((raster.height, raster.width), dtype=np.uint8)
    tiles = list(raster.tiles(tile_size, overlap))
    progress(f"  타일 {len(tiles)}개로 분할 처리 (타일 {tile_size}px, 겹침 {overlap}px)")

    for i, (read_win, core_win) in enumerate(tiles, start=1):
        rgb = raster.read_rgb(read_win)
        tile_label = seg_mod.build_label(rgb, cfg)

        r0, c0 = int(core_win.row_off), int(core_win.col_off)
        h, w = int(core_win.height), int(core_win.width)
        dr = int(read_win.row_off) + r0
        dc = int(read_win.col_off) + c0
        label[dr : dr + h, dc : dc + w] = tile_label[r0 : r0 + h, c0 : c0 + w]

        if i % 10 == 0 or i == len(tiles):
            progress(f"    타일 {i}/{len(tiles)}")

    return label


def run(
    image_path: str | Path,
    cfg: Config,
    tile_size: int = 4096,
    progress: Progress = lambda _m: None,
) -> Result:
    """정사영상 한 장을 끝까지 처리한다."""
    t0 = time.perf_counter()

    raster = open_raster(image_path, cfg.fallback_gsd_mm)
    progress(
        f"영상: {raster.width} x {raster.height} px ({raster.megapixels:.1f} MP), "
        f"GSD {raster.gsd_mm:.3f} mm/px"
    )
    progress(
        f"좌표계: {raster.crs} " if raster.georeferenced else "좌표계: 없음 (픽셀 좌표 사용)"
    )

    progress("색상 분리 중...")
    label = build_label_map(raster, cfg, tile_size, progress)

    segments: list[CrackSegment] = []
    dropped: dict[str, int] = {"짧음": 0, "뭉툭함": 0, "칠폭과다": 0}
    next_id = 1

    for idx, grade in enumerate(cfg.grades, start=1):
        progress(f"[{grade.id}] {grade.color} 처리 중...")
        mask = (label == idx).astype(np.uint8) * 255
        if not mask.any():
            progress(f"  {grade.color} 픽셀 없음")
            continue

        mask = seg_mod.remove_small_blobs(mask, cfg.denoise.min_blob_area_px)
        traced = skel_mod.trace_segments(
            mask,
            rdp_epsilon_px=cfg.vectorize.rdp_epsilon_px,
            prune_spur_px=cfg.vectorize.prune_spur_px,
        )

        kept = 0
        for t in traced:
            length_mm = t["length_px"] * raster.gsd_mm
            width_mm = t["paint_width_px"] * raster.gsd_mm

            if length_mm < cfg.denoise.min_length_mm:
                dropped["짧음"] += 1
                continue
            if width_mm > cfg.denoise.max_paint_width_mm:
                dropped["칠폭과다"] += 1
                continue
            elongation = length_mm / width_mm if width_mm > 1e-9 else float("inf")
            if elongation < cfg.denoise.min_elongation:
                dropped["뭉툭함"] += 1
                continue

            segments.append(
                CrackSegment(
                    id=next_id,
                    grade_id=grade.id,
                    grade_label=grade.label,
                    color=grade.color,
                    layer=grade.layer,
                    dxf_color=grade.dxf_color,
                    length_mm=length_mm,
                    paint_width_mm=width_mm,
                    points_world=raster.px_to_world(t["points_px"]),
                )
            )
            next_id += 1
            kept += 1

        progress(f"  세그먼트 {kept}개 채택 (후보 {len(traced)}개)")

    return Result(
        raster=raster,
        segments=segments,
        label=label,
        elapsed_s=time.perf_counter() - t0,
        dropped=dropped,
    )
