"""
Task 3 - Evaluate the classical segmentation (Task 2) against the SAM masks
(Task 1, treated as ground truth) on the held-out EVAL split only.

Produces:
  - report/eval_metrics_per_image.csv   (per-image IoU/Dice/precision/recall)
  - report/eval_summary.json            (aggregate stats)
  - report/assets/iou_histogram.png
  - report/assets/iou_vs_contrast.png
  - report/assets/worst_cases_grid.png  (qualitative: original | SAM GT | classical pred | error map)
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classical_pipeline import segment, contrast_score
from common import mask_iou, mask_dice

FINAL_CONFIG = dict(strategy="grabcut", clean=True, split_touching=True,
                     filter_components=True)


def precision_recall(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return float(precision), float(recall)


def pixel_accuracy(pred, gt):
    return float((pred.astype(bool) == gt.astype(bool)).mean())


def error_map(pred, gt):
    """RGB visualization: green=TP, red=FP (classical over-segmented),
    blue=FN (classical missed), black=TN."""
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    out[pred_b & gt_b] = (0, 200, 0)
    out[pred_b & ~gt_b] = (220, 30, 30)
    out[~pred_b & gt_b] = (30, 60, 220)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv", type=Path)
    ap.add_argument("--sam-masks-dir", default="outputs/task1_sam/masks", type=Path)
    ap.add_argument("--out-dir", default="report", type=Path)
    args = ap.parse_args()

    assets_dir = args.out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "eval"]
    print(f"Evaluating {len(rows)} eval images")

    per_image = []
    excluded = []
    for row in rows:
        gt_path = args.sam_masks_dir / f"{row['image_id']}.png"
        if not gt_path.exists():
            print(f"WARNING: missing SAM mask for {row['image_id']}, skipping")
            excluded.append(row["image_id"])
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127
        if gt.sum() == 0:
            # SAM's automatic mask generator found no egg-like region at all
            # (seen on the dev split for a few heavily foliage-occluded
            # photos) -- there is no usable pseudo-ground-truth to compare
            # against, so this image is excluded rather than silently scored
            # as a trivial IoU=0/1 case.
            print(f"NOTE: SAM found zero instances for {row['image_id']} "
                  f"(split={row['split']}), excluding from Task 3 metrics")
            excluded.append(row["image_id"])
            continue
        img = cv2.imread(row["path"])
        pred, _ = segment(img, **FINAL_CONFIG)
        pred_bool = pred > 127

        iou = mask_iou(pred_bool, gt)
        dice = mask_dice(pred_bool, gt)
        prec, rec = precision_recall(pred_bool, gt)
        acc = pixel_accuracy(pred_bool, gt)
        cscore = contrast_score(img)

        per_image.append({
            "image_id": row["image_id"], "path": row["path"],
            "iou": iou, "dice": dice, "precision": prec, "recall": rec,
            "pixel_accuracy": acc, "contrast_score": cscore,
            "gt_area_frac": float(gt.sum() / gt.size),
            "pred_area_frac": float(pred_bool.sum() / pred_bool.size),
        })

    per_image.sort(key=lambda r: r["iou"])

    csv_path = args.out_dir / "eval_metrics_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    print(f"Wrote {csv_path}")

    ious = [r["iou"] for r in per_image]
    dices = [r["dice"] for r in per_image]
    precs = [r["precision"] for r in per_image]
    recs = [r["recall"] for r in per_image]
    accs = [r["pixel_accuracy"] for r in per_image]

    summary = {
        "n_images": len(per_image),
        "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_precision": float(np.mean(precs)), "mean_recall": float(np.mean(recs)),
        "mean_pixel_accuracy": float(np.mean(accs)),
        "frac_iou_below_0.5": float(np.mean([i < 0.5 for i in ious])),
        "frac_iou_above_0.8": float(np.mean([i > 0.8 for i in ious])),
        "n_excluded_no_sam_gt": len(excluded),
        "excluded_image_ids": excluded,
    }
    with open(args.out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- plot: IoU histogram ---
    plt.figure(figsize=(6, 4))
    plt.hist(ious, bins=15, range=(0, 1), color="#4C72B0", edgecolor="white")
    plt.axvline(float(np.mean(ious)), color="#C44E52", linestyle="--", label=f"mean={np.mean(ious):.2f}")
    plt.xlabel("IoU (classical vs SAM)")
    plt.ylabel("# eval images")
    plt.title("Task 3: IoU distribution on eval split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "iou_histogram.png", dpi=140)
    plt.close()

    # --- plot: IoU vs contrast score ---
    cscores = [r["contrast_score"] for r in per_image]
    plt.figure(figsize=(6, 4))
    plt.scatter(cscores, ious, c="#55A868")
    plt.xlabel("Foreground/background contrast score (higher = easier)")
    plt.ylabel("IoU")
    plt.title("Task 3: segmentation quality vs. image contrast")
    plt.tight_layout()
    plt.savefig(assets_dir / "iou_vs_contrast.png", dpi=140)
    plt.close()

    # --- qualitative grid: worst N cases ---
    n_show = min(6, len(per_image))
    worst = per_image[:n_show]
    fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    col_titles = ["Original", "SAM (GT)", "Classical (pred)", "Error map (R=FP,B=FN,G=TP)"]
    for row_i, r in enumerate(worst):
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(args.sam_masks_dir / f"{r['image_id']}.png"), cv2.IMREAD_GRAYSCALE) > 127
        pred, _ = segment(cv2.imread(r["path"]), **FINAL_CONFIG)
        pred_bool = pred > 127
        err = error_map(pred_bool, gt)

        gt_overlay = img.copy()
        gt_overlay[gt] = (0.5 * np.array([255, 255, 0]) + 0.5 * gt_overlay[gt]).astype(np.uint8)
        pred_overlay = img.copy()
        pred_overlay[pred_bool] = (0.5 * np.array([255, 0, 255]) + 0.5 * pred_overlay[pred_bool]).astype(np.uint8)

        for col_i, im in enumerate([img, gt_overlay, pred_overlay, err]):
            ax = axes[row_i, col_i]
            ax.imshow(im)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=10)
        axes[row_i, 0].set_ylabel(f"IoU={r['iou']:.2f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(assets_dir / "worst_cases_grid.png", dpi=130)
    plt.close()

    print(f"Wrote qualitative assets to {assets_dir}")


if __name__ == "__main__":
    main()
