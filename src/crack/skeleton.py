"""세선화 -> 그래프화 -> 분기점 기준 세그먼트 분할 -> 폴리라인.

균열은 나뭇가지처럼 갈라진다. 갈래를 하나의 선으로 뭉뚱그리면 길이가
왜곡되므로, 분기점(junction)에서 끊어 각 가지를 독립 세그먼트로 만든다.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# 대각선보다 상하좌우를 먼저 보게 정렬하면 계단형 코너에서 경로가 덜 튄다.
_ORDERED_NEIGHBORS = sorted(_NEIGHBORS, key=lambda d: abs(d[0]) + abs(d[1]))


def _degree(skel: np.ndarray) -> np.ndarray:
    """각 스켈레톤 화소의 8-이웃 개수."""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return convolve(skel.astype(np.uint8), k, mode="constant", cval=0) * skel


def _order_chain(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """체인 성분의 화소들을 한쪽 끝에서 반대쪽 끝 순서로 정렬."""
    pts = set(coords)
    if len(pts) <= 2:
        return list(pts)

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        near = [(r + dr, c + dc) for dr, dc in _ORDERED_NEIGHBORS if (r + dr, c + dc) in pts]
        adj[(r, c)] = near

    ends = [p for p in pts if len(adj[p]) <= 1]
    start = min(ends) if ends else min(pts)

    order = [start]
    seen = {start}
    cur = start
    while True:
        nxt = next((q for q in adj[cur] if q not in seen), None)
        if nxt is None:
            break
        order.append(nxt)
        seen.add(nxt)
        cur = nxt
    return order


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker 폴리라인 단순화 (스택 기반, 재귀 없음)."""
    n = len(points)
    if n < 3 or epsilon <= 0:
        return points

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = points[i0], points[i1]
        seg = p1 - p0
        seg_len = float(np.hypot(*seg))
        sub = points[i0 + 1 : i1]
        if seg_len < 1e-12:
            dist = np.hypot(sub[:, 0] - p0[0], sub[:, 1] - p0[1])
        else:
            rel = sub - p0
            # 2차원 외적: numpy 2.x 는 np.cross 의 2D 입력을 더 이상 받지 않는다
            cross = seg[0] * rel[:, 1] - seg[1] * rel[:, 0]
            dist = np.abs(cross) / seg_len
        j = int(np.argmax(dist))
        if dist[j] > epsilon:
            idx = i0 + 1 + j
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))

    return points[keep]


def polyline_length(points: np.ndarray) -> float:
    """폴리라인을 따라간 누적 길이."""
    if len(points) < 2:
        return 0.0
    d = np.diff(points, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def trace_segments(
    binary: np.ndarray,
    rdp_epsilon_px: float = 1.5,
    prune_spur_px: int = 10,
) -> list[dict]:
    """이진 마스크에서 분기 단위 세그먼트들을 뽑는다.

    반환: [{'points_px': (N,2) float [col,row], 'length_px': float,
            'paint_width_px': float, 'free_ends': int}, ...]
    """
    binary = (binary > 0)
    if not binary.any():
        return []

    skel = skeletonize(binary)
    if not skel.any():
        return []

    # 칠 폭: 원본 마스크의 거리변환값 x2 (스켈레톤은 선의 중심선이므로)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)

    deg = _degree(skel)
    junction = skel & (deg >= 3)
    chain = skel & ~junction

    # 분기점 덩어리(보통 2~3화소)를 하나의 노드로 보고 중심을 구한다
    j_count, j_labels = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)
    j_centroid: dict[int, tuple[float, float]] = {}
    if j_count > 1:
        for jid in range(1, j_count):
            rs, cs = np.where(j_labels == jid)
            j_centroid[jid] = (float(rs.mean()), float(cs.mean()))

    c_count, c_labels = cv2.connectedComponents(chain.astype(np.uint8), connectivity=8)
    if c_count <= 1:
        return []

    ys, xs = np.where(c_labels > 0)
    by_comp: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(ys.tolist(), xs.tolist()):
        by_comp.setdefault(int(c_labels[r, c]), []).append((r, c))

    h, w = skel.shape
    raw: list[dict] = []

    for coords in by_comp.values():
        ordered = _order_chain(coords)
        if not ordered:
            continue

        # 양 끝에 붙어 있는 분기점을 이어 붙여야 길이가 정확해진다
        head_j = _adjacent_junction(ordered[0], j_labels, h, w)
        tail_j = _adjacent_junction(ordered[-1], j_labels, h, w)

        pts = [(float(r), float(c)) for r, c in ordered]
        if head_j is not None:
            pts.insert(0, j_centroid[head_j])
        if tail_j is not None:
            pts.append(j_centroid[tail_j])

        arr = np.array(pts, dtype=np.float64)          # (N,2) [row, col]
        simplified = _rdp(arr, rdp_epsilon_px)
        length_px = polyline_length(simplified)

        widths = dist[[r for r, _ in ordered], [c for _, c in ordered]]
        paint_width_px = float(np.median(widths)) * 2.0

        free_ends = int(head_j is None) + int(tail_j is None)
        raw.append(
            {
                "points_px": simplified[:, ::-1].copy(),   # [row,col] -> [col,row]
                "length_px": length_px,
                "paint_width_px": paint_width_px,
                "free_ends": free_ends,
            }
        )

    return _prune_spurs(raw, prune_spur_px)


def _adjacent_junction(
    pixel: tuple[int, int], j_labels: np.ndarray, h: int, w: int
) -> int | None:
    r, c = pixel
    for dr, dc in _NEIGHBORS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and j_labels[rr, cc] > 0:
            return int(j_labels[rr, cc])
    return None


def _prune_spurs(segments: list[dict], prune_spur_px: int) -> list[dict]:
    """분기점에서 뻗어 나온 짧은 잔가지를 버린다.

    양끝이 모두 자유단인 세그먼트(독립된 균열 하나)는 짧아도 살린다.
    길이 기준 필터는 뒤쪽 파이프라인에서 mm 단위로 따로 건다.
    """
    if prune_spur_px <= 0:
        return segments
    return [
        s
        for s in segments
        if not (s["free_ends"] == 1 and s["length_px"] < prune_spur_px)
    ]
