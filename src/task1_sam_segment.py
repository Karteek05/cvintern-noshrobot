"""
Task 1 - Segment eggs in every image using a pretrained SAM model.

SAM has no notion of "egg" by itself, so we run SamAutomaticMaskGenerator to
propose a large pool of class-agnostic masks per image, then score each
candidate with simple shape/position heuristics to decide which ones are
plausibly eggs:
  - convex (eggs are solid convex blobs -> high solidity = area / hull_area)
  - not a thin sliver and not the whole background (circularity + area bounds)
  - does not hug all four image borders (background / table masks do)
  - SAM's own predicted_iou and stability_score (mask quality)

This keeps Task 1 fully automatic (no manual per-image clicking) and supports
images with more than one egg (e.g. the two eggs boiling in a pot), since we
keep every candidate that passes the filters rather than only the single best
one.

These SAM outputs are later treated as pseudo-ground-truth for Task 3.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

MODEL_TYPE = "vit_b"


def load_generator(checkpoint: Path, device: str):
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        min_mask_region_area=400,
    )


def mask_shape_features(mask: np.ndarray):
    area = int(mask.sum())
    h, w = mask.shape
    ys, xs = np.where(mask)
    if area == 0:
        return None
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
    }


def is_egg_like(feat, sam_score):
    if feat is None:
        return False, 0.0
    if feat["touches_borders"] >= 2:
        return False, 0.0
    # Small, round-ish distractors (keyboard keys, tile/background fragments)
    # are common false positives with high solidity but are much smaller than
    # the actual egg and noticeably less circular; both bounds were picked by
    # inspecting candidate masks on a handful of hard dev images (see report).
    if not (0.025 <= feat["area_frac"] <= 0.55):
        return False, 0.0
    if feat["solidity"] < 0.88:
        return False, 0.0
    if feat["circularity"] < 0.75:
        return False, 0.0
    # Rejects SAM's coarser masks that merge two adjacent/touching eggs into
    # one elongated blob -- a real single egg's bbox is close to square
    # (aspect ratio ~1.1-1.6), while a merged two-egg blob is noticeably
    # longer along the axis the eggs are arranged on (observed ~2.8-3.0).
    if feat["aspect_ratio"] > 2.2:
        return False, 0.0
    score = (
        0.4 * sam_score
        + 0.3 * feat["solidity"]
        + 0.3 * feat["circularity"]
    )
    return True, score


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


def make_overlay(image_bgr, instance_masks, alpha=0.55):
    overlay = image_bgr.copy()
    rng = np.random.RandomState(0)
    colors = [tuple(int(c) for c in rng.randint(60, 255, 3)) for _ in instance_masks]
    for mask, color in zip(instance_masks, colors):
        colored = np.zeros_like(image_bgr)
        colored[:] = color
        overlay = np.where(mask[..., None], (alpha * colored + (1 - alpha) * overlay).astype(np.uint8), overlay)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def process_image(generator, image_path: Path):
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image_rgb)

    candidates = []
    for m in masks:
        seg = m["segmentation"]
        feat = mask_shape_features(seg)
        ok, score = is_egg_like(feat, m["predicted_iou"] * m["stability_score"])
        if ok:
            candidates.append({"mask": seg, "score": score, "feat": feat})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept = non_max_suppress(candidates, iou_thresh=0.7)
    kept = kept[:6]  # sanity cap; no image in this dataset has more than a couple eggs

    instance_masks = [c["mask"] for c in kept]
    combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for m in instance_masks:
        combined[m] = 255

    overlay = make_overlay(image_bgr, instance_masks)
    return combined, instance_masks, overlay, [c["feat"] for c in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv", type=Path)
    ap.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth", type=Path)
    ap.add_argument("--out-dir", default="outputs/task1_sam", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="process only first N images (debug)")
    ap.add_argument("--image-ids", nargs="*", default=None, help="process only these image_ids (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    generator = load_generator(args.checkpoint, device)

    masks_dir = args.out_dir / "masks"
    overlays_dir = args.out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    if args.image_ids:
        rows = [r for r in rows if r["image_id"] in args.image_ids]
    if args.limit:
        rows = rows[: args.limit]

    meta = {}
    for i, row in enumerate(rows):
        image_id = row["image_id"]
        image_path = Path(row["path"])
        print(f"[{i+1}/{len(rows)}] {image_id}")
        combined, instance_masks, overlay, feats = process_image(generator, image_path)

        cv2.imwrite(str(masks_dir / f"{image_id}.png"), combined)
        cv2.imwrite(str(overlays_dir / f"{image_id}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        meta[image_id] = {
            "num_instances": len(instance_masks),
            "instances": feats,
            "split": row["split"],
        }

    with open(args.out_dir / "masks_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_zero = sum(1 for v in meta.values() if v["num_instances"] == 0)
    print(f"Done. {len(meta)} images processed, {n_zero} with zero detected eggs.")


if __name__ == "__main__":
    main()
