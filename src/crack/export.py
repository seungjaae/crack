"""DXF / TSV 출력."""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf import units as ezunits

from .config import Config
from .model import CrackSegment
from .raster import Raster

TSV_COLUMNS = [
    "id",
    "grade",
    "grade_label",
    "color",
    "length_mm",
    "paint_width_mm",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "vertices",
    "wkt",
]


def _label_anchor(points: np.ndarray) -> tuple[float, float, float]:
    """폴리라인 중앙 지점과 그 지점의 진행 방향(도)을 구한다."""
    if len(points) < 2:
        return float(points[0, 0]), float(points[0, 1]), 0.0

    seg = np.diff(points, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = seg_len.sum()
    if total <= 0:
        return float(points[0, 0]), float(points[0, 1]), 0.0

    half = total / 2.0
    run = 0.0
    for i, ln in enumerate(seg_len):
        if run + ln >= half:
            t = (half - run) / ln if ln > 0 else 0.0
            x = points[i, 0] + t * seg[i, 0]
            y = points[i, 1] + t * seg[i, 1]
            angle = math.degrees(math.atan2(seg[i, 1], seg[i, 0]))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            return float(x), float(y), float(angle)
        run += ln

    return float(points[-1, 0]), float(points[-1, 1]), 0.0


def write_dxf(
    segments: list[CrackSegment],
    cfg: Config,
    raster: Raster,
    out_path: str | Path,
) -> Path:
    """색상(등급)별 레이어로 분리된 DXF 를 쓴다."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new(cfg.export.dxf_version, setup=True)
    doc.units = ezunits.MM if raster.mm_per_unit == 1.0 else ezunits.M
    msp = doc.modelspace()

    for grade in cfg.grades:
        if grade.layer not in doc.layers:
            doc.layers.add(name=grade.layer, color=grade.dxf_color)
        text_layer = grade.layer + cfg.export.text_layer_suffix
        if cfg.export.write_text and text_layer not in doc.layers:
            doc.layers.add(name=text_layer, color=grade.dxf_color)

    text_height = raster.mm_to_world_len(cfg.export.text_height_mm)

    for seg in segments:
        msp.add_lwpolyline(
            [(float(x), float(y)) for x, y in seg.points_world],
            dxfattribs={"layer": seg.layer},
        )
        if cfg.export.write_text:
            x, y, angle = _label_anchor(seg.points_world)
            msp.add_text(
                f"{seg.grade_id} L={seg.length_mm:.0f}mm",
                height=text_height,
                rotation=angle,
                dxfattribs={"layer": seg.layer + cfg.export.text_layer_suffix},
            ).set_placement((x, y + text_height * 0.3))

    doc.saveas(out)
    return out


def write_tsv(
    segments: list[CrackSegment],
    cfg: Config,
    out_path: str | Path,
) -> Path:
    """좌표/길이 표를 쓴다 (기본 탭 구분)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = cfg.export.tsv_delimiter

    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(d.join(TSV_COLUMNS) + "\n")
        for s in segments:
            sx, sy = s.start
            ex, ey = s.end
            row = [
                str(s.id),
                s.grade_id,
                s.grade_label,
                s.color,
                f"{s.length_mm:.1f}",
                f"{s.paint_width_mm:.1f}",
                f"{sx:.4f}",
                f"{sy:.4f}",
                f"{ex:.4f}",
                f"{ey:.4f}",
                str(len(s.points_world)),
                s.wkt(),
            ]
            fh.write(d.join(row) + "\n")
    return out


def summarize(segments: list[CrackSegment], cfg: Config) -> str:
    """등급별 개수/총연장 요약 문자열."""
    lines = ["등급별 집계", "-" * 52]
    total_n = 0
    total_len = 0.0
    for grade in cfg.grades:
        sel = [s for s in segments if s.grade_id == grade.id]
        length = sum(s.length_mm for s in sel)
        total_n += len(sel)
        total_len += length
        lines.append(
            f"  {grade.id} ({grade.color:<6} {grade.label})  "
            f"{len(sel):>5} 개   {length / 1000:>10.2f} m"
        )
    lines.append("-" * 52)
    lines.append(f"  {'합계':<24}{total_n:>7} 개   {total_len / 1000:>10.2f} m")
    return "\n".join(lines)


def write_preview(
    segments: list[CrackSegment],
    cfg: Config,
    raster: Raster,
    out_path: str | Path,
    max_px: int = 2000,
) -> Path:
    """검수용 미리보기 PNG. 원본 위에 추출된 선을 겹쳐 그린다."""
    import cv2

    from .raster import read_overview

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    img, scale = read_overview(raster, max_px)
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 원본은 어둡게 깔고 추출선을 밝게 올려서 대비를 준다
    canvas = (canvas * 0.45).astype(np.uint8)

    inv = ~raster.transform
    draw_bgr = {
        "red": (60, 60, 255),
        "yellow": (0, 220, 255),
        "cyan": (255, 220, 60),
    }

    for s in segments:
        cols_rows = np.array([inv * (float(x), float(y)) for x, y in s.points_world])
        pts = np.round(cols_rows * scale).astype(np.int32)
        bgr = draw_bgr.get(s.color, (255, 255, 255))
        cv2.polylines(canvas, [pts], isClosed=False, color=bgr, thickness=2,
                      lineType=cv2.LINE_AA)

    cv2.imwrite(str(out), canvas)
    return out
