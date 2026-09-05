"""색상 분리와 노이즈 제거.

콘크리트는 무채색(저채도)이고 락카는 고채도라서 HSV 공간에서
채도(S)만으로도 상당 부분 갈린다. 색상(H)으로 3색을 구분한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def color_mask(rgb: np.ndarray, hsv_ranges, cfg: Config) -> np.ndarray:
    """한 색상에 대한 이진 마스크(uint8 0/255)를 만든다."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h0, s0, v0, h1, s1, v1 in hsv_ranges:
        part = cv2.inRange(
            hsv,
            np.array([h0, s0, v0], dtype=np.uint8),
            np.array([h1, s1, v1], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, part)

    d = cfg.denoise
    if d.morph_open_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(d.morph_open_px))
    if d.morph_close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(d.morph_close_px))
    return mask


def build_label(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """타일 하나에 대해 등급 라벨맵을 만든다.

    0 = 배경, 1..N = cfg.grades 의 1-based 인덱스.
    """
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for idx, grade in enumerate(cfg.grades, start=1):
        spec = cfg.colors[grade.color]
        mask = color_mask(rgb, spec.hsv_ranges, cfg)
        label[mask > 0] = idx
    return label


def remove_small_blobs(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """면적이 작은 덩어리를 제거한다 (전체 영상 조립 후 1회 수행)."""
    if min_area_px <= 0:
        return mask
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area_px
    keep[0] = False
    return (keep[labels]).astype(np.uint8) * 255
