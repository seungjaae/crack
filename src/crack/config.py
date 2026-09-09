"""설정 로딩 및 검증."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class ColorSpec:
    name: str
    hsv_ranges: tuple[tuple[int, int, int, int, int, int], ...]


@dataclass(frozen=True)
class Grade:
    id: str
    color: str
    label: str
    min_mm: float
    max_mm: float  # 0 이면 상한 없음
    layer: str
    dxf_color: int


@dataclass(frozen=True)
class Denoise:
    morph_open_px: int
    morph_close_px: int
    min_blob_area_px: int
    min_length_mm: float
    min_elongation: float
    max_paint_width_mm: float


@dataclass(frozen=True)
class Vectorize:
    rdp_epsilon_px: float
    prune_spur_px: int


SORT_KEYS = ("grade_length", "grade", "length", "length_desc", "id")


@dataclass(frozen=True)
class Analysis:
    length_bins_mm: tuple[float, ...]
    sort_by: str


@dataclass(frozen=True)
class Export:
    dxf_version: str
    write_text: bool
    text_height_mm: float
    text_layer_suffix: str
    tsv_delimiter: str


@dataclass(frozen=True)
class Config:
    fallback_gsd_mm: float
    colors: dict[str, ColorSpec]
    grades: tuple[Grade, ...]
    denoise: Denoise
    vectorize: Vectorize
    export: Export
    analysis: Analysis

    def grade_for_color(self, color: str) -> Grade | None:
        for g in self.grades:
            if g.color == color:
                return g
        return None


def load(path: str | Path | None = None) -> Config:
    """TOML 설정을 읽어 Config 로 만든다."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {p}")

    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    colors: dict[str, ColorSpec] = {}
    for name, body in raw.get("colors", {}).items():
        ranges = body.get("hsv_ranges") or []
        if not ranges:
            raise ValueError(f"[colors.{name}] hsv_ranges 가 비어 있습니다")
        for r in ranges:
            if len(r) != 6:
                raise ValueError(
                    f"[colors.{name}] hsv_ranges 항목은 6개 값이어야 합니다: {r}"
                )
        colors[name] = ColorSpec(name=name, hsv_ranges=tuple(tuple(r) for r in ranges))

    grades = tuple(
        Grade(
            id=g["id"],
            color=g["color"],
            label=g["label"],
            min_mm=float(g["min_mm"]),
            max_mm=float(g["max_mm"]),
            layer=g["layer"],
            dxf_color=int(g["dxf_color"]),
        )
        for g in raw.get("grades", [])
    )
    if not grades:
        raise ValueError("[[grades]] 가 하나도 정의되지 않았습니다")

    seen_colors: set[str] = set()
    for g in grades:
        if g.color not in colors:
            raise ValueError(
                f"등급 {g.id} 가 참조하는 색상 '{g.color}' 이 [colors] 에 없습니다"
            )
        if g.color in seen_colors:
            raise ValueError(f"색상 '{g.color}' 이 둘 이상의 등급에 매핑되어 있습니다")
        seen_colors.add(g.color)

    # [analysis] 는 없어도 되도록 기본값을 둔다. 기존 설정 파일이 그대로 동작한다.
    a = raw.get("analysis", {})
    sort_by = str(a.get("sort_by", "grade_length"))
    if sort_by not in SORT_KEYS:
        raise ValueError(
            f"[analysis] sort_by '{sort_by}' 을 알 수 없습니다. "
            f"사용 가능: {', '.join(SORT_KEYS)}"
        )
    bins = sorted(float(x) for x in a.get("length_bins_mm", [300, 600, 1000, 2000]))
    if any(x <= 0 for x in bins):
        raise ValueError("[analysis] length_bins_mm 은 모두 0 보다 커야 합니다")

    return Config(
        fallback_gsd_mm=float(raw.get("project", {}).get("fallback_gsd_mm", 0.5)),
        colors=colors,
        grades=grades,
        denoise=Denoise(**raw["denoise"]),
        vectorize=Vectorize(**raw["vectorize"]),
        export=Export(**raw["export"]),
        analysis=Analysis(length_bins_mm=tuple(bins), sort_by=sort_by),
    )
