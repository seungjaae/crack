"""거더 자동 검출.

균열과 달리 거더는 색으로 구분할 수 없다. 거더와 바닥판이 같은 콘크리트라
경계에 색 차이가 없기 때문이다. 대신 거더는 기하가 매우 규칙적이다 —
전부 평행하고, 폭이 같고, 등간격으로 놓인다. 이 규칙성을 근거로 찾는다.

    1) 긴 직선들의 길이가중 평균으로 거더 방향을 소수점까지 구한다
    2) 그 방향이 세로가 되도록 회전한다
    3) 열별 '중앙값 밝기' 로 거더 띠를 찾는다 (합계가 아니라 중앙값이다.
       합계는 거더가 세로로 얼마나 길게 찍혔는지에 휘둘린다)
    4) 찾은 띠들에 등간격 격자를 맞춘다. 격자는 가려진 거더의 위치도 알려준다
    5) 길이 범위는 확신도 높은 거더들의 합의값을 쓴다. 한 경간의 거더는
       끝단이 거의 정렬돼 있기 때문이다
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .raster import Raster


@dataclass
class Girder:
    """검출된 거더 하나. 네 꼭짓점으로 표현한다."""

    id: int
    points_px: np.ndarray      # (4,2) 원본 픽셀 좌표
    points_world: np.ndarray   # (4,2) 실좌표
    width_mm: float
    length_mm: float
    support: float             # 지지 밝기. 낮으면 가려졌을 가능성이 있다

    def wkt(self) -> str:
        ring = list(self.points_world) + [self.points_world[0]]
        body = ", ".join(f"{x:.4f} {y:.4f}" for x, y in ring)
        return f"POLYGON(({body}))"


def dominant_angle(gray: np.ndarray, cfg: Config) -> float | None:
    """긴 직선들의 길이가중 평균으로 거더 방향(도)을 구한다.

    정수로 반올림하면 안 된다. 거더가 1,700px 길면 0.3도 오차가 끝단에서
    10px 밀림으로 커진다.
    """
    g = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(g, cfg.girder.canny_lo, cfg.girder.canny_hi)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720,
        threshold=cfg.girder.hough_threshold,
        minLineLength=cfg.girder.hough_min_len,
        maxLineGap=cfg.girder.hough_max_gap,
    )
    if lines is None or len(lines) < 8:
        return None

    seg = lines.reshape(-1, 4).astype(float)
    d = seg[:, 2:] - seg[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    ln = np.hypot(d[:, 0], d[:, 1])

    # 최빈 방향을 길이가중 히스토그램으로 잡고
    hist = np.zeros(180)
    np.add.at(hist, ang.astype(int) % 180, ln)
    hist = cv2.GaussianBlur(hist.reshape(1, -1).astype(np.float32), (1, 9), 0).ravel()
    coarse = float(np.argmax(hist))

    # 그 주변만 모아 원형평균으로 정밀화 (각도는 180도 주기라 2배각을 쓴다)
    near = np.abs((ang - coarse + 90) % 180 - 90) < cfg.girder.angle_tol_deg
    if near.sum() < 8:
        return None
    th = np.radians(ang[near] * 2)
    lw = ln[near]
    return float(
        np.degrees(np.arctan2((lw * np.sin(th)).sum(), (lw * np.cos(th)).sum())) / 2 % 180
    )


def _rotate_to_vertical(img: np.ndarray, angle: float, shape: tuple[int, int]):
    """거더가 세로가 되도록 회전. (회전행렬, 출력크기) 를 함께 돌려준다."""
    h, w = shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle - 90, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=0), M, (nw, nh)


def _column_profile(rot: np.ndarray, valid: np.ndarray, chunks: int) -> np.ndarray:
    """열별 중앙값 밝기. 세로를 여러 구간으로 나눠 그 중 최대를 취한다.

    일부만 화면에 걸친 거더는 전체 중앙값을 내면 주변 바닥에 희석된다.
    구간별 최대를 쓰면 어느 한 구간에서만 밝아도 잡힌다.
    """
    nh, nw = rot.shape
    rows = np.where(valid.sum(axis=1) > nw * 0.5)[0]
    if len(rows) < 50:
        return np.zeros(nw)
    r0, r1 = rows.min(), rows.max()

    bounds = np.linspace(r0, r1, chunks + 1).astype(int)
    acc = np.zeros((chunks, nw))
    for c in range(chunks):
        a, b = bounds[c], bounds[c + 1]
        sub, sv = rot[a:b], valid[a:b]
        need = (b - a) * 0.4
        for x in range(nw):
            m = sv[:, x]
            if m.sum() > need:
                acc[c, x] = np.median(sub[:, x][m])
    prof = acc.max(axis=0)
    return cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (1, 7), 0).ravel()


def _runs_above(values: np.ndarray, thr: float, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(list(values > thr) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def _band_profile(rot, valid, c: float, half: float) -> np.ndarray | None:
    """거더 한 줄을 따라간 세로 방향 밝기."""
    nh, nw = rot.shape
    a, b = int(round(c - half)), int(round(c + half))
    if a < 0 or b >= nw or b <= a:
        return None
    need = (b - a) * 0.6
    col = np.zeros(nh)
    for r in range(nh):
        m = valid[r, a:b]
        if m.sum() > need:
            col[r] = np.median(rot[r, a:b][m])
    return cv2.GaussianBlur(col.reshape(-1, 1).astype(np.float32), (1, 15), 0).ravel()


def detect(raster: Raster, cfg: Config, gray: np.ndarray | None = None) -> list[Girder]:
    """정사영상에서 거더를 찾는다."""
    if gray is None:
        gray = cv2.cvtColor(raster.read_rgb(), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    gc = cfg.girder

    angle = dominant_angle(gray, cfg)
    if angle is None:
        return []

    rot, M, (nw, nh) = _rotate_to_vertical(gray.astype(np.float32), angle, (h, w))
    valid = _rotate_to_vertical(
        np.full((h, w), 255, np.uint8), angle, (h, w)
    )[0] > 0
    Minv = cv2.invertAffineTransform(M)

    prof = _column_profile(rot, valid, gc.chunks)
    nzv = prof[prof > 0]
    if nzv.size < 50:
        return []
    thr, _ = cv2.threshold(
        nzv.astype(np.uint8).reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    runs = _runs_above(prof, thr, gc.min_run_px)
    if not runs:
        return []

    widths = np.array([b - a for a, b in runs], dtype=float)
    med_w = float(np.median(widths))
    half = med_w / 2

    # 폭이 정상인 띠만으로 등간격 격자를 맞춘다.
    # 너무 넓은 띠는 거더와 틈새가 붙은 것이라 기준으로 쓸 수 없다.
    clean = [(a, b) for a, b in runs if abs((b - a) - med_w) <= med_w * 0.35]
    if len(clean) >= 2:
        cxs = np.array([(a + b) / 2 for a, b in clean])
        k = np.arange(len(cxs))
        pitch, phase = np.linalg.lstsq(
            np.vstack([k, np.ones(len(k))]).T, cxs, rcond=None
        )[0]
    else:
        pitch, phase = med_w * 2.2, (runs[0][0] + runs[0][1]) / 2

    if pitch <= med_w:
        return []

    # 확신도 높은 거더들의 길이 범위를 합의값으로 삼는다
    spans = []
    for i in range(len(clean)):
        col = _band_profile(rot, valid, pitch * i + phase, half)
        if col is None:
            continue
        best = max(_runs_above(col, gc.extent_brightness, 40),
                   key=lambda t: t[1] - t[0], default=None)
        if best and best[1] - best[0] > nh * gc.min_length_ratio:
            spans.append(best)
    if not spans:
        return []
    y0 = int(np.median([s[0] for s in spans]))
    y1 = int(np.median([s[1] for s in spans]))

    out: list[Girder] = []
    i = 0
    while True:
        c = pitch * i + phase
        if c + half + gc.width_pad_px >= nw:
            break
        if c - half - gc.width_pad_px < 0:
            i += 1
            continue
        col = _band_profile(rot, valid, c, half)
        if col is None:
            i += 1
            continue
        band = col[y0:y1]
        sup = float(np.median(band[band > 0])) if (band > 0).any() else 0.0
        if sup < gc.support_brightness:
            i += 1
            continue

        # 한 경간의 거더는 길이가 거의 같다. 자체 측정이 합의값보다 크게
        # 짧으면 가려진 것이므로 자체 측정을 믿지 않는다.
        own = max(_runs_above(col, gc.extent_brightness, 40),
                  key=lambda t: t[1] - t[0], default=None)
        consensus = y1 - y0
        if own and (own[1] - own[0]) >= consensus * gc.trust_own_extent:
            yy0, yy1 = own
        else:
            yy0, yy1 = y0, y1

        x0 = c - half - gc.width_pad_px
        x1 = c + half + gc.width_pad_px
        quad = np.array([[x0, yy0], [x1, yy0], [x1, yy1], [x0, yy1]], np.float32)
        px = cv2.transform(quad.reshape(-1, 1, 2), Minv).reshape(-1, 2)

        out.append(
            Girder(
                id=len(out) + 1,
                points_px=px,
                points_world=raster.px_to_world(px),
                width_mm=(x1 - x0) * raster.gsd_mm,
                length_mm=(yy1 - yy0) * raster.gsd_mm,
                support=sup,
            )
        )
        i += 1

    return out
