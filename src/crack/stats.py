"""추출된 균열의 속성값 분석.

폭별(=등급별) 분류, 길이별 구간 분류, 총연장 집계, 정렬을 담당한다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .config import SORT_KEYS, Config
from .model import CrackSegment


@dataclass
class GradeStat:
    """폭 등급 하나에 대한 집계."""

    grade_id: str
    label: str
    color: str
    count: int
    total_mm: float
    min_mm: float
    max_mm: float
    mean_mm: float
    median_mm: float
    pct_count: float      # 전체 개수 대비 %
    pct_length: float     # 전체 연장 대비 %


@dataclass
class LengthBin:
    """길이 구간 하나에 대한 집계."""

    label: str
    lo_mm: float | None       # None = 하한 없음
    hi_mm: float | None       # None = 상한 없음
    count: int
    total_mm: float
    pct_count: float
    by_grade: dict[str, int] = field(default_factory=dict)   # 등급별 개수


@dataclass
class Summary:
    count: int
    total_mm: float
    mean_mm: float
    median_mm: float
    min_mm: float
    max_mm: float
    by_grade: list[GradeStat]
    by_length: list[LengthBin]

    @property
    def total_m(self) -> float:
        return self.total_mm / 1000.0


# ---------------------------------------------------------------- 정렬
def sort_segments(
    segments: list[CrackSegment], cfg: Config, key: str | None = None
) -> list[CrackSegment]:
    """정렬 기준에 맞춰 새 리스트를 돌려준다 (원본은 건드리지 않는다)."""
    key = key or cfg.analysis.sort_by
    if key not in SORT_KEYS:
        raise ValueError(
            f"정렬 기준 '{key}' 을 알 수 없습니다. 사용 가능: {', '.join(SORT_KEYS)}"
        )

    order = {g.id: i for i, g in enumerate(cfg.grades)}

    if key == "id":
        return sorted(segments, key=lambda s: s.id)
    if key == "length":
        return sorted(segments, key=lambda s: s.length_mm)
    if key == "length_desc":
        return sorted(segments, key=lambda s: -s.length_mm)
    if key == "grade":
        return sorted(segments, key=lambda s: (order.get(s.grade_id, 99), s.id))
    return sorted(segments, key=lambda s: (order.get(s.grade_id, 99), -s.length_mm))


# ---------------------------------------------------------------- 집계
def _bin_labels(bounds: list[float]) -> list[tuple[str, float | None, float | None]]:
    """경계값 목록에서 (라벨, 하한, 상한) 구간들을 만든다."""
    if not bounds:
        return [("전체", None, None)]
    b = sorted(float(x) for x in bounds)
    out: list[tuple[str, float | None, float | None]] = [
        (f"{b[0]:,.0f} mm 미만", None, b[0])
    ]
    for lo, hi in zip(b, b[1:]):
        out.append((f"{lo:,.0f} ~ {hi:,.0f} mm", lo, hi))
    out.append((f"{b[-1]:,.0f} mm 이상", b[-1], None))
    return out


def analyze(segments: list[CrackSegment], cfg: Config) -> Summary:
    """세그먼트 목록에서 속성값 집계를 만든다."""
    lengths = [s.length_mm for s in segments]
    total = sum(lengths)
    n = len(segments)

    by_grade: list[GradeStat] = []
    for grade in cfg.grades:
        sel = [s.length_mm for s in segments if s.grade_id == grade.id]
        g_total = sum(sel)
        by_grade.append(
            GradeStat(
                grade_id=grade.id,
                label=grade.label,
                color=grade.color,
                count=len(sel),
                total_mm=g_total,
                min_mm=min(sel) if sel else 0.0,
                max_mm=max(sel) if sel else 0.0,
                mean_mm=statistics.fmean(sel) if sel else 0.0,
                median_mm=statistics.median(sel) if sel else 0.0,
                pct_count=len(sel) / n * 100 if n else 0.0,
                pct_length=g_total / total * 100 if total else 0.0,
            )
        )

    by_length: list[LengthBin] = []
    for label, lo, hi in _bin_labels(cfg.analysis.length_bins_mm):
        # 구간은 [하한, 상한) 으로 잡아 경계값이 두 구간에 겹치지 않게 한다
        sel = [
            s
            for s in segments
            if (lo is None or s.length_mm >= lo) and (hi is None or s.length_mm < hi)
        ]
        counts: dict[str, int] = {}
        for s in sel:
            counts[s.grade_id] = counts.get(s.grade_id, 0) + 1
        by_length.append(
            LengthBin(
                label=label,
                lo_mm=lo,
                hi_mm=hi,
                count=len(sel),
                total_mm=sum(s.length_mm for s in sel),
                pct_count=len(sel) / n * 100 if n else 0.0,
                by_grade=counts,
            )
        )

    return Summary(
        count=n,
        total_mm=total,
        mean_mm=statistics.fmean(lengths) if lengths else 0.0,
        median_mm=statistics.median(lengths) if lengths else 0.0,
        min_mm=min(lengths) if lengths else 0.0,
        max_mm=max(lengths) if lengths else 0.0,
        by_grade=by_grade,
        by_length=by_length,
    )


# ---------------------------------------------------------------- 리포트
def _w(s: str) -> int:
    """한글·한자는 콘솔에서 두 칸을 차지한다. 실제 표시 폭을 센다."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int, right: bool = False) -> str:
    gap = max(0, n - _w(s))
    return (" " * gap + s) if right else (s + " " * gap)


def format_report(summary: Summary, cfg: Config) -> str:
    """사람이 읽는 속성값 분석 리포트."""
    if summary.count == 0:
        return "추출된 균열이 없습니다."

    L: list[str] = []
    bar = "=" * 74
    L.append(bar)
    L.append(" 균열 속성값 분석")
    L.append(bar)
    L.append(f"  총 세그먼트   {summary.count:,} 개")
    L.append(f"  총 연장       {summary.total_m:,.2f} m   ({summary.total_mm:,.0f} mm)")
    L.append(f"  평균 / 중앙값 {summary.mean_mm:,.1f} mm / {summary.median_mm:,.1f} mm")
    L.append(f"  최소 / 최대   {summary.min_mm:,.1f} mm / {summary.max_mm:,.1f} mm")

    # ---- 폭별(등급별) ----
    L.append("")
    L.append("─ 폭별 분류 (색상 = 균열 폭 등급) " + "─" * 40)
    head = (
        _pad("  등급", 8) + _pad("색상", 10) + _pad("기준 폭", 24)
        + _pad("개수", 8, True) + _pad("연장(m)", 12, True) + _pad("연장비율", 10, True)
    )
    L.append(head)
    L.append("  " + "-" * 70)
    for g in summary.by_grade:
        L.append(
            _pad(f"  {g.grade_id}", 8)
            + _pad(g.color, 10)
            + _pad(g.label, 24)
            + _pad(f"{g.count:,}", 8, True)
            + _pad(f"{g.total_mm / 1000:,.2f}", 12, True)
            + _pad(f"{g.pct_length:.1f}%", 10, True)
        )
    L.append("  " + "-" * 70)
    L.append(
        _pad("  합계", 42)
        + _pad(f"{summary.count:,}", 8, True)
        + _pad(f"{summary.total_m:,.2f}", 12, True)
        + _pad("100.0%", 10, True)
    )

    # ---- 길이별 ----
    grade_ids = [g.id for g in cfg.grades]
    L.append("")
    L.append("─ 길이별 분류 " + "─" * 59)
    head = _pad("  구간", 22) + _pad("개수", 8, True) + _pad("연장(m)", 12, True) + _pad("비율", 9, True)
    for gid in grade_ids:
        head += _pad(gid, 7, True)
    L.append(head)
    L.append("  " + "-" * 70)
    for b in summary.by_length:
        row = (
            _pad(f"  {b.label}", 22)
            + _pad(f"{b.count:,}", 8, True)
            + _pad(f"{b.total_mm / 1000:,.2f}", 12, True)
            + _pad(f"{b.pct_count:.1f}%", 9, True)
        )
        for gid in grade_ids:
            row += _pad(str(b.by_grade.get(gid, 0)), 7, True)
        L.append(row)

    return "\n".join(L)
