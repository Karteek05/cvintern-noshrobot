"""
Task 4 - Segment + track the two eggs in the YouTube demo video
(https://www.youtube.com/watch?v=Y4JyV94y2_A, "Raw and Boiled Egg Spinning").

The clip has an intro title card (0-9s) and a text-slide explanation
(42-60s) with no eggs in frame; only 9-42s shows the actual eggs on the
table, so that's the range we process (see report for how this was found:
by eye, scanning a contact sheet of frames every 2s).

Segmentation here deliberately differs from the Task 2 dataset pipeline (the
task brief explicitly allows this): the camera is static and a bright
checkered shirt in the upper part of the frame defeats a single global
brightness threshold (it's brighter than the eggs), so we crop a fixed
table-only ROI before running the same Otsu-Lab-L + watershed pipeline.
GrabCut (Task 2's best performer) was not used here because it is ~5s/image,
which is impractical across ~800 video frames.

Tracking: a lightweight greedy centroid tracker (src/tracker.py) capped at 2
simultaneous tracks (the known egg count for this clip), so an occasional
false-positive blob (the eggs' reflection on the glossy table) cannot spawn a
spurious third identity once both real eggs are already tracked. A short
missed-frame allowance lets a track survive brief occlusion by the hand.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from classical_pipeline import segment
from tracker import CentroidTracker

SEG_CONFIG = dict(strategy="otsu_lab_l", clean=True, split_touching=True,
                   filter_components=True, roi_y0=100, use_texture_filter=False)

TRACK_COLORS = {1: (60, 200, 60), 2: (60, 120, 230), 3: (200, 60, 200), 4: (60, 200, 200)}


def detections_from_instances(instances, max_keep=2):
    scored = []
    for m in instances:
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        area = int(m.sum())
        cx, cy = float(xs.mean()), float(ys.mean())
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        scored.append((area, (cx, cy), bbox))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [(c, b) for _, c, b in scored[:max_keep]]


def draw_tracks(frame, tracks, trail_len=20, show_missed_grace=12):
    """show_missed_grace: once a track has gone unmatched for longer than
    this many frames, stop drawing it -- the tracker keeps it alive
    internally for much longer (to re-link the same ID after the egg
    reappears), but freezing a box in place on screen for a multi-second
    hand occlusion would just look like a bug rather than a design choice."""
    out = frame.copy()
    for t in tracks:
        if t.frames_since_update > show_missed_grace:
            continue
        color = TRACK_COLORS.get(t.track_id, (255, 255, 255))
        x0, y0, x1, y1 = t.bbox
        is_predicted = t.frames_since_update > 0
        thickness = 1 if is_predicted else 3
        cv2.ellipse(out, (int((x0 + x1) / 2), int((y0 + y1) / 2)),
                     (int((x1 - x0) / 2), int((y1 - y0) / 2)), 0, 0, 360, color, thickness)
        label = f"egg {t.track_id}" + (" (occluded)" if is_predicted else "")
        cv2.putText(out, label, (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        pts = [c for _, c in t.history[-trail_len:]]
        for i in range(1, len(pts)):
            p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p1 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(out, p0, p1, color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video/source.mp4", type=Path)
    ap.add_argument("--out", default="outputs/task4_tracking/tracked.mp4", type=Path)
    ap.add_argument("--start-sec", type=float, default=9.0)
    ap.add_argument("--end-sec", type=float, default=42.0)
    ap.add_argument("--max-tracks", type=int, default=2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f, end_f = int(args.start_sec * fps), int(args.end_sec * fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = CentroidTracker(max_tracks=args.max_tracks, max_distance=100, max_missed=190)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    n_both, n_one, n_zero = 0, 0, 0
    while frame_idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        _, instances = segment(frame, **SEG_CONFIG)
        dets = detections_from_instances(instances, max_keep=args.max_tracks)
        if len(dets) == 2:
            n_both += 1
        elif len(dets) == 1:
            n_one += 1
        else:
            n_zero += 1

        tracks = tracker.update(dets, frame_idx)
        out_frame = draw_tracks(frame, tracks)
        cv2.putText(out_frame, f"t={frame_idx/fps:.1f}s", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(out_frame)
        frame_idx += 1

    writer.release()
    cap.release()
    n_total = n_both + n_one + n_zero
    print(f"Wrote {args.out}")
    print(f"Per-frame detections over {n_total} frames: "
          f"both_eggs={n_both} ({n_both/n_total:.0%}), "
          f"one_egg={n_one} ({n_one/n_total:.0%}), "
          f"zero_eggs={n_zero} ({n_zero/n_total:.0%})")
    print(f"Total distinct track IDs created: {tracker._next_id - 1}")


if __name__ == "__main__":
    main()
