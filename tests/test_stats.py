"""속성값 분석: 폭별/길이별 분류, 총연장, 정렬."""

import csv
import io

import numpy as np
import pytest

from crack import config as config_mod
from crack import export, stats
from crack.model import CrackSegment


@pytest.fixture(scope="module")
def cfg():
    return config_mod.load()


def seg(sid: int, grade_id: str, color: str, length_mm: float) -> CrackSegment:
    return CrackSegment(
        id=sid,
        grade_id=grade_id,
        grade_label=f"{grade_id} 라벨",
        color=color,
        layer=f"CRACK_{grade_id}",
        dxf_color=1,
        length_mm=length_mm,
        paint_width_mm=10.0,
        points_world=np.array([[0.0, 0.0], [length_mm, 0.0]]),
    )


@pytest.fixture
def sample():
    return [
        seg(1, "W1", "red", 250.0),
        seg(2, "W1", "red", 1500.0),
        seg(3, "W2", "yellow", 300.0),     # 구간 경계값
        seg(4, "W2", "yellow", 800.0),
        seg(5, "W3", "cyan", 2500.0),
        seg(6, "W3", "cyan", 150.0),
    ]


# ------------------------------------------------------------------ 집계
def test_total_length_is_the_sum(cfg, sample):
    s = stats.analyze(sample, cfg)
    assert s.count == 6
    assert s.total_mm == pytest.approx(5500.0)
    assert s.total_m == pytest.approx(5.5)


def test_width_grouping_is_by_grade(cfg, sample):
    s = stats.analyze(sample, cfg)
    by_id = {g.grade_id: g for g in s.by_grade}
    assert by_id["W1"].count == 2
    assert by_id["W1"].total_mm == pytest.approx(1750.0)
    assert by_id["W2"].total_mm == pytest.approx(1100.0)
    assert by_id["W3"].total_mm == pytest.approx(2650.0)
    # 연장 비율의 합은 100%
    assert sum(g.pct_length for g in s.by_grade) == pytest.approx(100.0)


def test_empty_grade_still_reported(cfg):
    """한 색상이 하나도 안 나와도 표에서 빠지면 안 된다."""
    s = stats.analyze([seg(1, "W1", "red", 500.0)], cfg)
    assert len(s.by_grade) == len(cfg.grades)
    assert {g.grade_id for g in s.by_grade} == {g.id for g in cfg.grades}
    assert [g.count for g in s.by_grade if g.grade_id == "W3"] == [0]


def test_length_bins_partition_everything(cfg, sample):
    s = stats.analyze(sample, cfg)
    assert sum(b.count for b in s.by_length) == len(sample)
    assert sum(b.total_mm for b in s.by_length) == pytest.approx(s.total_mm)


def test_boundary_value_goes_to_upper_bin(cfg, sample):
    """경계값(300mm)은 '300 미만'이 아니라 '300~600'에 들어가야 한다."""
    s = stats.analyze(sample, cfg)
    bins = {b.label: b for b in s.by_length}
    under = next(b for b in s.by_length if b.hi_mm == 300 and b.lo_mm is None)
    assert under.count == 2                      # 250, 150
    over = next(b for b in s.by_length if b.lo_mm == 300)
    assert over.count == 1                       # 300 (경계값)


def test_length_bin_crosstab_by_grade(cfg, sample):
    s = stats.analyze(sample, cfg)
    top = next(b for b in s.by_length if b.hi_mm is None)   # 2000mm 이상
    assert top.count == 1
    assert top.by_grade == {"W3": 1}


def test_analyze_handles_no_segments(cfg):
    s = stats.analyze([], cfg)
    assert s.count == 0 and s.total_mm == 0
    assert len(s.by_grade) == len(cfg.grades)


# ------------------------------------------------------------------ 정렬
def test_sort_by_grade_then_length_desc(cfg, sample):
    out = stats.sort_segments(sample, cfg, "grade_length")
    assert [s.grade_id for s in out] == ["W1", "W1", "W2", "W2", "W3", "W3"]
    assert [s.length_mm for s in out[:2]] == [1500.0, 250.0]


def test_sort_by_length(cfg, sample):
    asc = [s.length_mm for s in stats.sort_segments(sample, cfg, "length")]
    assert asc == sorted(asc)
    desc = [s.length_mm for s in stats.sort_segments(sample, cfg, "length_desc")]
    assert desc == sorted(desc, reverse=True)


def test_sort_does_not_mutate_input(cfg, sample):
    before = [s.id for s in sample]
    stats.sort_segments(sample, cfg, "length_desc")
    assert [s.id for s in sample] == before


def test_unknown_sort_key_is_rejected(cfg, sample):
    with pytest.raises(ValueError, match="알 수 없"):
        stats.sort_segments(sample, cfg, "nope")


# ------------------------------------------------------------------ 출력
def test_report_contains_the_headline_numbers(cfg, sample):
    text = stats.format_report(stats.analyze(sample, cfg), cfg)
    assert "5.50 m" in text          # 총연장
    assert "폭별 분류" in text
    assert "길이별 분류" in text
    for g in cfg.grades:
        assert g.id in text


def test_stats_tsv_round_trip(cfg, sample, tmp_path):
    summary = stats.analyze(sample, cfg)
    p = export.write_stats_tsv(summary, cfg, tmp_path / "s.tsv")
    rows = list(csv.DictReader(io.open(p, encoding="utf-8-sig"), delimiter="\t"))

    sections = {r["구분"] for r in rows}
    assert sections == {"전체", "폭별", "길이별"}

    total = next(r for r in rows if r["구분"] == "전체")
    assert float(total["총연장_m"]) == pytest.approx(5.5)
    assert int(total["개수"]) == 6

    width_rows = [r for r in rows if r["구분"] == "폭별"]
    assert len(width_rows) == len(cfg.grades)
    assert sum(int(r["개수"]) for r in width_rows) == 6

    length_rows = [r for r in rows if r["구분"] == "길이별"]
    assert sum(int(r["개수"]) for r in length_rows) == 6


def test_custom_bins_are_honoured(cfg, sample):
    from dataclasses import replace

    c2 = replace(cfg, analysis=replace(cfg.analysis, length_bins_mm=(1000.0,)))
    s = stats.analyze(sample, c2)
    assert len(s.by_length) == 2
    assert s.by_length[0].count == 4       # 1000 미만: 250, 300, 800, 150
    assert s.by_length[1].count == 2       # 1000 이상: 1500, 2500
