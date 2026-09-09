import json

import pytest

from crack import config as config_mod
from crack import export, pipeline


@pytest.fixture(scope="module")
def result(synthetic_deck):
    cfg = config_mod.load()
    return cfg, pipeline.run(synthetic_deck, cfg)


def test_reads_georeferencing(result):
    _cfg, res = result
    assert res.raster.georeferenced
    assert res.raster.crs.to_epsg() == 5186
    assert res.raster.gsd_mm == pytest.approx(0.5, abs=1e-6)


def test_finds_every_crack_and_nothing_else(result, synthetic_deck):
    _cfg, res = result
    truth = json.loads(
        synthetic_deck.with_suffix(".truth.json").read_text(encoding="utf-8")
    )
    assert len(res.segments) == len(truth)   # 오검출 0, 미검출 0

    got = {}
    for s in res.segments:
        got[s.color] = got.get(s.color, 0) + 1
    assert got == {"red": 2, "yellow": 1, "cyan": 3}


def test_total_length_matches_truth(result, synthetic_deck):
    _cfg, res = result
    truth = json.loads(
        synthetic_deck.with_suffix(".truth.json").read_text(encoding="utf-8")
    )
    want = sum(t["length_mm"] for t in truth)
    got = sum(s.length_mm for s in res.segments)
    assert abs(got - want) / want < 0.02


def test_paint_width_is_measured(result):
    _cfg, res = result
    for s in res.segments:
        assert 8.0 < s.paint_width_mm < 14.0   # 실제 10mm


def test_exports(result, tmp_path):
    cfg, res = result
    dxf = export.write_dxf(res.segments, cfg, res.raster, tmp_path / "a.dxf")
    tsv = export.write_tsv(res.segments, cfg, tmp_path / "a.tsv")
    assert dxf.is_file() and tsv.is_file()

    import ezdxf

    doc = ezdxf.readfile(dxf)
    layers = {l.dxf.name for l in doc.layers}
    for g in cfg.grades:
        assert g.layer in layers                      # 색상별 레이어 분리
    msp = doc.modelspace()
    assert len(msp.query("LWPOLYLINE")) == len(res.segments)
    assert len(msp.query("TEXT")) == len(res.segments)   # 도면에 길이 텍스트 병기

    header, *rows = tsv.read_text(encoding="utf-8-sig").strip().split("\n")
    assert header.split("\t") == export.TSV_COLUMNS
    assert len(rows) == len(res.segments)


def test_per_color_previews_are_written(result, tmp_path):
    """색상별로 한 장씩, 그 색 세그먼트만 담아 저장한다."""
    cfg, res = result
    paths = export.write_previews_by_color(
        res.segments, cfg, res.raster, tmp_path, "deck"
    )
    assert len(paths) == len(cfg.grades)
    for p, g in zip(paths, cfg.grades):
        assert p.is_file() and p.stat().st_size > 0
        assert g.id in p.name and g.color in p.name
