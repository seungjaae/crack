"""검증용 합성 정사영상 생성기.

실제 드론 데이터가 오기 전에 파이프라인을 끝까지 돌려보고
길이 측정 정확도를 정답과 비교하기 위한 것이다.

콘크리트 질감 + 현장 노이즈(그림자/타이어자국/흙/낙엽) 위에
정답 좌표를 아는 3색 균열선을 그린다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

GSD_M = 0.0005          # 0.5 mm/px
PAINT_WIDTH_PX = 20     # 10 mm 폭 락카선
W, H = 4000, 3000       # 2.0m x 1.5m

# BGR 이 아니라 RGB 로 다룬다
PAINT_RGB = {
    "red": (222, 38, 32),
    "yellow": (247, 205, 40),
    "cyan": (40, 178, 226),
}

# 정답 균열 (색상, 폴리라인 정점들). cyan 은 Y자 분기를 3세그먼트로 표현.
TRUTH = [
    ("red", [(300, 400), (1500, 900)]),
    ("red", [(2600, 300), (3700, 1400)]),
    ("yellow", [(400, 2400), (1400, 2000), (2300, 2450)]),
    ("cyan", [(1800, 1500), (2400, 1900)]),          # 줄기
    ("cyan", [(2400, 1900), (3100, 1750)]),          # 가지 A
    ("cyan", [(2400, 1900), (3000, 2600)]),          # 가지 B
]


def concrete_background(rng: np.random.Generator) -> np.ndarray:
    """회색 콘크리트 질감."""
    base = rng.normal(168, 14, (H, W)).astype(np.float32)
    coarse = cv2.resize(
        rng.normal(0, 26, (H // 24, W // 24)).astype(np.float32), (W, H),
        interpolation=cv2.INTER_CUBIC,
    )
    tex = np.clip(base + coarse, 40, 235)
    img = np.repeat(tex[:, :, None], 3, axis=2)
    img[:, :, 0] *= 1.01     # 아주 옅은 색편차
    img[:, :, 2] *= 0.99
    return np.clip(img, 0, 255).astype(np.uint8)


def add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """현장 노이즈: 그림자, 타이어 자국, 흙 얼룩, 낙엽."""
    out = img.copy()

    # 그림자 (넓고 어두운 사각 영역)
    shadow = np.ones((H, W), np.float32)
    cv2.rectangle(shadow, (2200, 1900), (3900, 2950), 0.62, -1)
    shadow = cv2.GaussianBlur(shadow, (201, 201), 0)
    out = (out * shadow[:, :, None]).astype(np.uint8)

    # 타이어 자국 (어두운 회색 굵은 줄) - 선형이라 가장 헷갈리는 노이즈
    for y in (620, 700):
        cv2.line(out, (0, y), (W, y + 130), (70, 70, 74), 46, cv2.LINE_AA)

    # 흙 얼룩
    for _ in range(90):
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.circle(out, c, int(rng.integers(6, 34)), (105, 96, 84), -1, cv2.LINE_AA)

    # 낙엽 - 갈색/주황 계열이라 red/yellow 임계값을 실제로 위협한다
    for _ in range(45):
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        ax, ay = int(rng.integers(22, 55)), int(rng.integers(14, 34))
        col = (int(rng.integers(120, 175)), int(rng.integers(58, 105)), int(rng.integers(20, 52)))
        cv2.ellipse(out, (cx, cy), (ax, ay), float(rng.integers(0, 180)),
                    0, 360, col, -1, cv2.LINE_AA)

    return out


def draw_cracks(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for color, pts in TRUTH:
        arr = np.array(pts, np.int32)
        cv2.polylines(out, [arr], False, PAINT_RGB[color], PAINT_WIDTH_PX, cv2.LINE_AA)
    return out


def truth_table() -> list[dict]:
    rows = []
    for i, (color, pts) in enumerate(TRUTH, start=1):
        a = np.array(pts, np.float64)
        d = np.diff(a, axis=0)
        length_px = float(np.hypot(d[:, 0], d[:, 1]).sum())
        rows.append(
            {
                "id": i,
                "color": color,
                "length_px": round(length_px, 2),
                "length_mm": round(length_px * GSD_M * 1000, 2),
                "points_px": pts,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="검증용 합성 정사영상 생성")
    ap.add_argument("-o", "--out", default="tests/fixtures/synthetic_deck.tif")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img = draw_cracks(add_noise(concrete_background(rng), rng))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    transform = from_origin(200000.0, 550000.0, GSD_M, GSD_M)
    with rasterio.open(
        out, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
        crs=CRS.from_epsg(5186), transform=transform,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    ) as ds:
        ds.write(np.transpose(img, (2, 0, 1)))

    truth = truth_table()
    tp = out.with_suffix(".truth.json")
    tp.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(r["length_mm"] for r in truth)
    print(f"생성: {out}  ({W}x{H}px, GSD {GSD_M*1000}mm, EPSG:5186)")
    print(f"정답: {tp}  세그먼트 {len(truth)}개, 총연장 {total/1000:.3f} m")
    for r in truth:
        print(f"  #{r['id']} {r['color']:<7} {r['length_mm']:>9.1f} mm")


if __name__ == "__main__":
    main()
