import textwrap

import pytest

from crack import config as config_mod


def test_default_config_loads():
    cfg = config_mod.load()
    assert len(cfg.grades) == 3
    assert {g.id for g in cfg.grades} == {"W1", "W2", "W3"}
    # 사용자 확정 기준: 0.1~0.2 / 0.2~0.3 / 0.3~
    by_id = {g.id: g for g in cfg.grades}
    assert (by_id["W1"].min_mm, by_id["W1"].max_mm) == (0.1, 0.2)
    assert (by_id["W2"].min_mm, by_id["W2"].max_mm) == (0.2, 0.3)
    assert by_id["W3"].min_mm == 0.3 and by_id["W3"].max_mm == 0.0


def test_grade_color_mapping_is_configurable():
    """색상 순서는 현장 규칙에 따라 바뀔 수 있어야 한다."""
    cfg = config_mod.load()
    assert cfg.grade_for_color("red").id == "W1"
    assert cfg.grade_for_color("cyan").id == "W3"
    assert cfg.grade_for_color("magenta") is None


def _write(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


BASE_TAIL = """
    [denoise]
    morph_open_px = 3
    morph_close_px = 5
    min_blob_area_px = 200
    min_length_mm = 30.0
    min_elongation = 3.0
    max_paint_width_mm = 60.0
    [vectorize]
    rdp_epsilon_px = 1.5
    prune_spur_px = 10
    [export]
    dxf_version = "R2010"
    write_text = true
    text_height_mm = 50.0
    text_layer_suffix = "_TEXT"
    tsv_delimiter = "\t"
"""


def test_rejects_unknown_color(tmp_path):
    p = _write(tmp_path, """
    [colors.red]
    hsv_ranges = [[0, 80, 60, 10, 255, 255]]
    [[grades]]
    id = "W1"
    color = "purple"
    label = "x"
    min_mm = 0.1
    max_mm = 0.2
    layer = "L"
    dxf_color = 1
    """ + BASE_TAIL)
    with pytest.raises(ValueError, match="purple"):
        config_mod.load(p)


def test_rejects_duplicate_color(tmp_path):
    p = _write(tmp_path, """
    [colors.red]
    hsv_ranges = [[0, 80, 60, 10, 255, 255]]
    [[grades]]
    id = "W1"
    color = "red"
    label = "x"
    min_mm = 0.1
    max_mm = 0.2
    layer = "L1"
    dxf_color = 1
    [[grades]]
    id = "W2"
    color = "red"
    label = "y"
    min_mm = 0.2
    max_mm = 0.3
    layer = "L2"
    dxf_color = 2
    """ + BASE_TAIL)
    with pytest.raises(ValueError, match="둘 이상"):
        config_mod.load(p)
