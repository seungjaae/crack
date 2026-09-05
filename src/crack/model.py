"""파이프라인이 만들어 내보내는 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrackSegment:
    """분기 단위로 쪼개진 균열 하나."""

    id: int
    grade_id: str
    grade_label: str
    color: str
    layer: str
    dxf_color: int
    length_mm: float
    paint_width_mm: float
    points_world: np.ndarray  # (N,2) [x, y] 실좌표

    @property
    def start(self) -> tuple[float, float]:
        return float(self.points_world[0, 0]), float(self.points_world[0, 1])

    @property
    def end(self) -> tuple[float, float]:
        return float(self.points_world[-1, 0]), float(self.points_world[-1, 1])

    def wkt(self) -> str:
        body = ", ".join(f"{x:.4f} {y:.4f}" for x, y in self.points_world)
        return f"LINESTRING({body})"
