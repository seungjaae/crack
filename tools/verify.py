"""추출 결과를 합성 정답과 대조한다."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_tsv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    truth = json.loads(Path("tests/fixtures/synthetic_deck.truth.json").read_text("utf-8"))
    got = load_tsv(Path("out/synthetic_deck.tsv"))

    # 정답 중점과 가장 가까운 추출 세그먼트를 짝지어 비교 (좌표는 EPSG:5186)
    ox, oy, gsd = 200000.0, 550000.0, 0.0005

    def px_to_world(pt):
        return ox + pt[0] * gsd, oy - pt[1] * gsd

    def mid_world(pts):
        a = np.array([px_to_world(p) for p in pts])
        return a.mean(axis=0)

    rows = []
    used: set[int] = set()
    for t in truth:
        tm = mid_world(t["points_px"])
        best, best_d = None, 1e18
        for i, g in enumerate(got):
            if i in used or g["color"] != t["color"]:
                continue
            gm = np.array([
                (float(g["start_x"]) + float(g["end_x"])) / 2,
                (float(g["start_y"]) + float(g["end_y"])) / 2,
            ])
            d = float(np.hypot(*(gm - tm)))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            rows.append((t, None, None))
            continue
        used.add(best)
        rows.append((t, got[best], best_d))

    print(f"{'정답ID':<7}{'색상':<9}{'정답(mm)':>11}{'측정(mm)':>11}{'오차':>9}{'오차%':>8}{'칠폭mm':>9}")
    print("-" * 66)
    errs = []
    for t, g, _d in rows:
        if g is None:
            print(f"{t['id']:<7}{t['color']:<9}{t['length_mm']:>11.1f}{'미검출':>11}")
            errs.append(None)
            continue
        meas = float(g["length_mm"])
        err = meas - t["length_mm"]
        pct = err / t["length_mm"] * 100
        errs.append(pct)
        print(f"{t['id']:<7}{t['color']:<9}{t['length_mm']:>11.1f}{meas:>11.1f}"
              f"{err:>+9.1f}{pct:>+7.2f}%{float(g['paint_width_mm']):>9.1f}")

    print("-" * 66)
    ok = [e for e in errs if e is not None]
    print(f"검출: {len(ok)}/{len(truth)}   "
          f"평균 절대오차 {np.mean(np.abs(ok)):.2f}%   최대 {np.max(np.abs(ok)):.2f}%")
    print(f"오검출(정답에 없는 선): {len(got) - len(used)} 개")

    fail = len(ok) < len(truth) or np.max(np.abs(ok)) > 5.0
    print("\n결과:", "실패" if fail else "통과 (전건 검출, 최대오차 5% 이내)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
