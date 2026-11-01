"""
Build a unified image manifest from the Roboflow export and split it into
dev/eval sets (4:1) while guarding against data leakage.

The Roboflow folder export (train/valid/test) is not a meaningful split for
us -- it was built for a classification model, and because the source photos
look like short bursts (same egg, same background, consecutive shots), visually
near-duplicate frames can land on both sides of *any* naive split. That is the
"data leakage" the task description warns about: if a near-duplicate of a dev
image ends up in eval, eval performance is inflated because the model (or in
our case, the classical algorithm being hand-tuned on dev) has effectively
already seen that frame.

We guard against this by:
  1. Pooling all 110 images together (ignoring the original train/valid/test
     folders).
  2. Computing a perceptual hash (pHash) per image and clustering images whose
     hash Hamming distance is small (near-duplicate bursts) into groups.
  3. Assigning whole groups -- never individual images -- to dev or eval, so
     near-duplicates always land on the same side.
  4. Greedily filling eval to ~1/5 of the images by assigning groups (largest
     first) until the target count is reached, which keeps the ratio close to
     4:1 despite the group-size constraint.
"""
import argparse
import csv
import json
from pathlib import Path

import imagehash
from PIL import Image

HASH_THRESHOLD = 6  # max Hamming distance to consider two images near-duplicates


def find_images(raw_dir: Path):
    # Roboflow puts everything under <split>/<class>/*.jpg; a couple of images
    # landed in "Unlabeled" instead of "egg" during annotation, but they are
    # still egg photos, so include every class subfolder.
    return sorted(raw_dir.glob("*/*/*.jpg"))


def cluster_near_duplicates(records):
    """Union-find clustering on pHash Hamming distance."""
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["hash"] - records[j]["hash"] <= HASH_THRESHOLD:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--out-manifest", default="data/manifest.csv", type=Path)
    ap.add_argument("--out-splits-dir", default="data/splits", type=Path)
    ap.add_argument("--eval-fraction", default=0.2, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    images = find_images(args.raw_dir)
    print(f"Found {len(images)} images")

    records = []
    for p in images:
        im = Image.open(p)
        h = imagehash.phash(im, hash_size=16)
        records.append({
            "path": p,
            "roboflow_split": p.parent.parent.name,  # train/valid/test
            "hash": h,
        })

    groups = cluster_near_duplicates(records)
    groups.sort(key=len, reverse=True)
    print(f"Clustered into {len(groups)} near-duplicate groups "
          f"(threshold={HASH_THRESHOLD}); "
          f"largest group size={len(groups[0])}")

    n_total = len(records)
    n_eval_target = round(n_total * args.eval_fraction)

    # Deterministic shuffle of groups (by seed) before greedy fill, so which
    # groups land in eval isn't an artifact of pHash/file ordering.
    import random
    rng = random.Random(args.seed)
    group_order = list(range(len(groups)))
    rng.shuffle(group_order)

    eval_idx = set()
    eval_count = 0
    for gi in group_order:
        members = groups[gi]
        if eval_count >= n_eval_target:
            break
        for idx in members:
            eval_idx.add(idx)
        eval_count += len(members)

    for i, rec in enumerate(records):
        rec["group_id"] = next(gi for gi, g in enumerate(groups) if i in g)
        rec["split"] = "eval" if i in eval_idx else "dev"

    n_eval = sum(1 for r in records if r["split"] == "eval")
    n_dev = n_total - n_eval
    print(f"dev={n_dev} eval={n_eval} (target eval~{n_eval_target}, "
          f"ratio dev:eval = {n_dev / n_eval:.2f}:1)")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_splits_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "path", "roboflow_split", "group_id", "phash", "split"])
        for i, rec in enumerate(records):
            image_id = rec["path"].stem
            w.writerow([image_id, rec["path"].as_posix(), rec["roboflow_split"],
                        rec["group_id"], str(rec["hash"]), rec["split"]])

    for split in ("dev", "eval"):
        with open(args.out_splits_dir / f"{split}.txt", "w") as f:
            for rec in records:
                if rec["split"] == split:
                    f.write(rec["path"].as_posix() + "\n")

    multi_image_groups = [g for g in groups if len(g) > 1]
    print(f"{len(multi_image_groups)} groups had >1 near-duplicate image "
          f"(kept together on one side of the split)")
    with open(args.out_splits_dir / "groups_debug.json", "w") as f:
        json.dump(
            {str(gi): [records[i]["path"].name for i in members]
             for gi, members in enumerate(groups) if len(members) > 1},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
