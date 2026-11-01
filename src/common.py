"""Shared mask-geometry utilities used by both the SAM pipeline (Task 1) and
the classical image-processing pipeline (Task 2), so "does this blob look
like an egg" is scored the same way in both places.
"""
import cv2
import numpy as np


def mask_shape_features(mask: np.ndarray):
    """mask: 2D bool/uint8 array. Returns None if mask is empty."""
    area = int(mask.sum())
    if area == 0:
        return None
    h, w = mask.shape
    ys, xs = np.where(mask)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = (cv2.contourArea(c) / hull_area) if hull_area > 0 else 0
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    touches_borders = sum([x0 <= 1, y0 <= 1, x1 >= w - 2, y1 >= h - 2])
    bbox_w, bbox_h = (x1 - x0 + 1), (y1 - y0 + 1)
    aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
    return {
        "area_frac": float(area / (h * w)),
        "solidity": float(solidity),
        "circularity": float(min(circularity, 1.0)),
        "touches_borders": int(touches_borders),
        "aspect_ratio": float(aspect_ratio),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "contour": c,
    }


def non_max_suppress(candidates, iou_thresh=0.7):
    """candidates: list of dicts with 'mask' and 'score', sorted desc by score."""
    kept = []
    for cand in candidates:
        dup = False
        for k in kept:
            inter = np.logical_and(cand["mask"], k["mask"]).sum()
            union = np.logical_or(cand["mask"], k["mask"]).sum()
            if union > 0 and inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else (1.0 if a.sum() == b.sum() == 0 else 0.0)


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0
