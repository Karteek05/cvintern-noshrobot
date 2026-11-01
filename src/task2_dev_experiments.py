"""
Task 2 experiment harness: run several classical-segmentation configs over the
*dev* split only, scoring each against the Task 1 SAM masks (treated as
pseudo-ground-truth during development -- the eval split is held out for the
unbiased Task 3 report). Produces report/experiment_log_results.csv, which the
PDF report's experiment log table is built from.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from classical_pipeline import segment
from common import mask_iou, mask_dice

TRIALS = [
    dict(name="T1_otsu_gray_raw", strategy="otsu_gray", clean=False,
         split_touching=False, filter_components=False,
         note="Baseline: global Otsu on grayscale, no cleanup at all."),

    dict(name="T2_otsu_gray_clean", strategy="otsu_gray", clean=True,
         split_touching=False, filter_components=True,
         note="Add morphological open/close + hole fill + shape filter."),

    dict(name="T3_otsu_lab_l_clean", strategy="otsu_lab_l", clean=True,
         split_touching=False, filter_components=True,
         note="Swap grayscale for Lab L channel (illumination-focused)."),

    dict(name="T4_hsv_sv_clean", strategy="hsv_sv", clean=True,
         split_touching=False, filter_components=True,
         note="Low-saturation & high-value mask, targeting white eggshell."),

    dict(name="T5_adaptive_gray_clean", strategy="adaptive_gray", clean=True,
         split_touching=False, filter_components=True,
         note="Local adaptive threshold instead of a single global Otsu split."),

    dict(name="T6_canny_fill_clean", strategy="canny_fill", clean=True,
         split_touching=False, filter_components=True,
         note="Edge-based: Canny + closing + contour fill, for boundary-only cases."),

    dict(name="T7_grabcut_clean", strategy="grabcut", clean=True,
         split_touching=False, filter_components=True,
         note="Classical GrabCut seeded with a fixed center-biased rectangle."),

    dict(name="T8_lab_l_grabcut_fallback", strategy="otsu_lab_l", clean=True,
         split_touching=False, filter_components=True, grabcut_fallback_thresh=18.0,
         note="Hybrid: fast Otsu-Lab-L path, falls back to GrabCut when the "
              "Otsu between-class contrast is below a learned threshold."),

    dict(name="T9_final_with_watershed", strategy="otsu_lab_l", clean=True,
         split_touching=True, filter_components=True, grabcut_fallback_thresh=18.0,
         note="Same as T8 but with watershed splitting enabled, to separate "
              "touching/adjacent eggs (e.g. two eggs in a pot) into instances."),

    dict(name="T10_grabcut_watershed", strategy="grabcut", clean=True,
         split_touching=True, filter_components=True,
         note="T7 (best so far) + watershed splitting, to check whether "
              "multi-egg images (e.g. the boiling pot) need instance splitting."),

    dict(name="T11_grabcut_mask_init", strategy="grabcut_mask_init", clean=True,
         split_touching=True, filter_components=True,
         note="Replace T7's fixed center rectangle with an Otsu-Lab-L-derived "
              "probable-foreground mask as the GrabCut seed, so it adapts to "
              "eggs that are off-center or a different size than usual."),

    dict(name="T12_grabcut_multiseed", strategy="grabcut_multiseed", clean=True,
         split_touching=True, filter_components=True,
         note="Follow-up after inspecting Task 3's worst cases: T10's single "
              "fixed center rectangle misses eggs that lie outside it (e.g. "
              "3 eggs spread across the full frame width). Seed GrabCut once "
              "per Otsu-Lab-L connected component instead of once globally."),
]


def load_gt_mask(image_id, sam_masks_dir: Path):
    p = sam_masks_dir / f"{image_id}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return m > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv", type=Path)
    ap.add_argument("--sam-masks-dir", default="outputs/task1_sam/masks", type=Path)
    ap.add_argument("--out-csv", default="report/experiment_log_results.csv", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="trial name(s) to run (default: all)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    if args.limit:
        rows = rows[: args.limit]

    trials = TRIALS
    if args.only:
        trials = [t for t in TRIALS if t["name"] in args.only]

    existing = {}
    if args.out_csv.exists():
        with open(args.out_csv) as f:
            for r in csv.DictReader(f):
                existing[r["trial"]] = r

    print(f"Evaluating {len(rows)} dev images x {len(trials)} trial(s)")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for trial in trials:
        kwargs = {k: v for k, v in trial.items() if k not in ("name", "note")}
        ious, dices = [], []
        n_missing_gt = 0
        t0 = time.time()
        for row in rows:
            gt = load_gt_mask(row["image_id"], args.sam_masks_dir)
            if gt is None:
                n_missing_gt += 1
                continue
            img = cv2.imread(row["path"])
            pred, _ = segment(img, **kwargs)
            pred_bool = pred > 127
            ious.append(mask_iou(pred_bool, gt))
            dices.append(mask_dice(pred_bool, gt))
        elapsed = time.time() - t0
        mean_iou = float(np.mean(ious)) if ious else float("nan")
        mean_dice = float(np.mean(dices)) if dices else float("nan")
        print(f"{trial['name']:28s} mean_IoU={mean_iou:.3f} mean_Dice={mean_dice:.3f} "
              f"(n={len(ious)}, missing_gt={n_missing_gt}, {elapsed:.1f}s)")
        existing[trial["name"]] = {
            "trial": trial["name"], "strategy": trial.get("strategy"),
            "mean_iou": mean_iou, "mean_dice": mean_dice,
            "n": len(ious), "note": trial["note"],
        }

    # preserve the original TRIALS ordering regardless of what subset ran
    all_names = [t["name"] for t in TRIALS]
    ordered = [existing[n] for n in all_names if n in existing]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "strategy", "mean_iou", "mean_dice", "n", "note"])
        w.writeheader()
        w.writerows(ordered)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
