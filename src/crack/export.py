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
from .stats import Summary

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


STATS_COLUMNS = [
    "구분", "항목", "설명", "개수", "총연장_mm", "총연장_m", "개수비율_%", "연장비율_%",
]


def write_stats_tsv(
    summary: Summary,
    cfg: Config,
    out_path: str | Path,
) -> Path:
    """속성값 분석 결과를 표로 쓴다. 엑셀에서 바로 열린다."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = cfg.export.tsv_delimiter
    grade_ids = [g.id for g in cfg.grades]

    def row(vals) -> str:
        return d.join(str(v) for v in vals) + "\n"

    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(row(STATS_COLUMNS + grade_ids))

        # 전체 합계
        totals = {g.grade_id: g.count for g in summary.by_grade}
        fh.write(row([
            "전체", "TOTAL", "총합",
            summary.count, f"{summary.total_mm:.1f}", f"{summary.total_m:.3f}",
            "100.0", "100.0",
        ] + [totals.get(gid, 0) for gid in grade_ids]))

        # 폭별(등급별)
        for g in summary.by_grade:
            fh.write(row([
                "폭별", g.grade_id, g.label,
                g.count, f"{g.total_mm:.1f}", f"{g.total_mm / 1000:.3f}",
                f"{g.pct_count:.1f}", f"{g.pct_length:.1f}",
            ] + [g.count if gid == g.grade_id else 0 for gid in grade_ids]))

        # 길이별
        for b in summary.by_length:
            pct_len = b.total_mm / summary.total_mm * 100 if summary.total_mm else 0.0
            key = f"{b.lo_mm or 0:.0f}-{b.hi_mm:.0f}" if b.hi_mm else f"{b.lo_mm:.0f}+"
            fh.write(row([
                "길이별", key, b.label,
                b.count, f"{b.total_mm:.1f}", f"{b.total_mm / 1000:.3f}",
                f"{b.pct_count:.1f}", f"{pct_len:.1f}",
            ] + [b.by_grade.get(gid, 0) for gid in grade_ids]))

    return out
