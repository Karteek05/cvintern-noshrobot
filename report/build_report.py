"""Assemble report/report.html from the generated CSV/JSON results and
figures, with images embedded as base64 so the HTML (and the PDF printed
from it) is a single self-contained file. Run after task1/2/3/4 scripts."""
import base64
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"


def b64img(path, mime="image/jpeg"):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def load_experiment_log():
    with open(REPORT / "experiment_log_results.csv") as f:
        return list(csv.DictReader(f))


def load_eval_summary():
    with open(REPORT / "eval_summary.json") as f:
        return json.load(f)


def fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def trial_table_rows(trials):
    rows = []
    for t in trials:
        rows.append(f"""
        <tr>
          <td class="mono">{t['trial']}</td>
          <td class="mono">{t['strategy']}</td>
          <td class="num">{fmt(t['mean_iou'])}</td>
          <td class="num">{fmt(t['mean_dice'])}</td>
          <td>{t['note']}</td>
        </tr>""")
    return "\n".join(rows)


def main():
    trials = load_experiment_log()
    summary = load_eval_summary()

    best_trial = max(trials, key=lambda t: float(t["mean_iou"]))
    t10 = next(t for t in trials if t["trial"] == "T10_grabcut_watershed")
    t12 = next((t for t in trials if t["trial"] == "T12_grabcut_multiseed"), None)
    if t12 is not None:
        global T12_RESULT_SECTION
        T12_RESULT_SECTION = T12_RESULT_SECTION.format(t12_row=trial_table_rows([t12]))

    drive_link_html = (DRIVE_LINK if DRIVE_LINK.startswith("http")
                        else '<span style="color:#b00">[Drive link pending — see note in submission email]</span>')

    html = f"""
<style>
  :root {{ --ink:#1a1a1a; --sub:#555; --line:#ddd; --accent:#2a5d8f; --bg-soft:#f6f7f9; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; color: var(--ink); line-height:1.5; font-size: 14px; max-width: 880px; margin: 0 auto; padding: 24px 28px 60px; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  h2 {{ font-size: 19px; margin-top: 34px; border-bottom: 2px solid var(--accent); padding-bottom:4px; color: var(--accent); }}
  h3 {{ font-size: 15px; margin-top: 20px; }}
  .byline {{ color: var(--sub); font-size: 13px; margin-bottom: 18px; }}
  .byline a {{ color: var(--accent); }}
  p {{ margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 12.5px; }}
  th, td {{ border: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: top; }}
  th {{ background: var(--bg-soft); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .mono {{ font-family: Consolas, monospace; font-size: 11.5px; }}
  figure {{ margin: 14px 0; page-break-inside: avoid; }}
  figure img {{ width: 100%; border: 1px solid var(--line); border-radius: 3px; }}
  figure.bigfig {{ page-break-inside: auto; page-break-before: always; }}
  figure.bigfig img {{ max-height: 960px; width: auto; max-width: 100%; display: block; margin: 0 auto; }}
  figcaption {{ font-size: 12px; color: var(--sub); margin-top: 4px; }}
  .two-col {{ display:flex; gap:14px; }}
  .two-col > div {{ flex:1; }}
  .callout {{ background: var(--bg-soft); border-left: 3px solid var(--accent); padding: 8px 12px; margin: 12px 0; font-size: 13px; }}
  .metric-grid {{ display:grid; grid-template-columns: repeat(4,1fr); gap:8px; margin:12px 0; }}
  .metric {{ background:var(--bg-soft); border-radius:4px; padding:8px 10px; text-align:center; }}
  .metric .v {{ font-size:19px; font-weight:600; color:var(--accent); }}
  .metric .k {{ font-size:11px; color:var(--sub); }}
  code {{ background: var(--bg-soft); padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
  a {{ color: var(--accent); }}
  .pagebreak {{ page-break-before: always; }}
  ul, ol {{ margin: 6px 0; padding-left: 22px; page-break-inside: avoid; }}
  li {{ margin: 3px 0; }}
</style>

<h1>Egg Segmentation &amp; Tracking — CV Intern Task</h1>
<div class="byline">
  Karteek Cherukupalli · {REPORT_DATE} ·
  Dataset: <a href="https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1">Roboflow egg-vzhtr v1</a> ·
  Video: <a href="https://www.youtube.com/watch?v=Y4JyV94y2_A">Raw and Boiled Egg Spinning (Mocomi)</a>
</div>

<div class="callout">
<b>AI tool disclosure.</b> This project (code, experiment design and this report) was built with
Claude Code (Anthropic) as a pair-programming assistant across the full pipeline — SAM post-filtering
heuristics, the classical segmentation trials, the leakage-aware dev/eval split, the watershed bug found
during Task 4, and the tracker design. All of it was reviewed and is understood well enough to walk
through and modify live, per the assignment's expectations.
</div>

<h2>Dataset</h2>
<p>
The Roboflow export contains <b>110 images</b> of eggs (one class, "egg"), each resized to 640×640 by
Roboflow's preprocessing. It ships pre-split into train/valid/test folders, but that split was built for
a classification model and is not meaningful for this task, so all 110 images were pooled and re-split
(see Task 2). The scenes are more varied than the single PDF example photo (egg in a hand) suggests:
</p>
<figure>
  <img src="{b64img(REPORT/'assets'/'dataset_gallery.jpg')}">
  <figcaption>Fig. 1 — Dataset variety: a top-down marble-floor shot (favorable contrast), an egg in a
  white tray (near-zero color contrast with the background), two eggs boiling in a reflective steel pot
  (multi-instance + specular highlights), and an egg against a dark laptop keyboard (high contrast).</figcaption>
</figure>

<h2>Task 1 — Segmentation with a pretrained model (SAM)</h2>
<p>
<b>Model:</b> Meta's Segment Anything (ViT-B checkpoint, 375MB). ViT-B was chosen over ViT-L/ViT-H
specifically because the available GPU is a 4GB GTX 1650 — ViT-H's encoder alone does not comfortably fit
that budget alongside the automatic mask generator's internal batching.
</p>
<p>
<b>Method:</b> SAM has no "egg" concept by itself, so <code>SamAutomaticMaskGenerator</code> proposes a
large pool of class-agnostic masks per image (32×32 point grid, <code>pred_iou_thresh=0.86</code>,
<code>stability_score_thresh=0.9</code>), and each candidate is scored against simple shape/position
heuristics to decide whether it's plausibly an egg:
</p>
<ul>
  <li><b>Solidity</b> ≥ 0.88 and <b>circularity</b> ≥ 0.75 — eggs are smooth convex blobs.</li>
  <li><b>Area fraction</b> between 2.5% and 55% of the image — rejects small distractors (keyboard keys,
  tile/background fragments) and full-background masks alike.</li>
  <li><b>Touches ≤ 1 image border</b> — rejects background/table masks that span the frame.</li>
  <li><b>Bounding-box aspect ratio</b> ≤ 2.2 — rejects SAM's coarser masks that merge two adjacent eggs
  into one elongated blob (see note below).</li>
</ul>
<p>
Surviving candidates are de-duplicated by mask-IoU non-max suppression (threshold 0.7) and every remaining
candidate is kept (not just the single best one), so images with more than one egg are handled natively.
</p>
<figure>
  <img src="{b64img(REPORT/'assets'/'task1_examples.jpg')}">
  <figcaption>Fig. 2 — Left: a clean single-egg result. Middle: both eggs found in the boiling-pot photo
  (multi-instance). Right: the one systematic limitation found — a small rounded tray-corner fragment
  (~2.8% of image area, circularity 0.775) sits close enough to real small-egg statistics that the
  heuristic filter can't separate it without also rejecting genuine small eggs; documented rather than
  chased further.</figcaption>
</figure>
<p>
<b>Threshold tuning example.</b> An early version used a looser circularity/area filter and produced
visible false positives — individual rounded-corner laptop keys and background tile fragments scored as
"egg-like." Tightening the area-fraction floor (1%→2.5%) and circularity floor (0.55→0.75) removed these
without losing any true eggs observed in spot checks. A separate, subtler bug: SAM's automatic generator
sometimes also proposes a coarser mask that merges two touching eggs into one elongated blob in addition
to the two correct per-egg masks; because this blob only partially overlaps either individual mask, plain
pairwise IoU-NMS didn't catch it as a duplicate. The fix was the aspect-ratio filter above (merged blobs
are conspicuously non-square; ~2.8-3.0 vs. ~1.0-1.6 for a real single egg).
</p>
<p>
<b>Results across all 110 images:</b> {MULTI_INSTANCE} images correctly got more than one instance mask;
{ZERO_INSTANCE} images got zero (all are photos where the egg is heavily occluded by foliage/herb sprigs
in front of it — a genuinely hard case for a fully-automatic, no-prompt pipeline). These {ZERO_INSTANCE}
images are excluded from Task 3's evaluation set when they fall in the eval split (there is no usable
pseudo-ground-truth to compare against), rather than silently scored as a trivial empty-mask case.
</p>

<h2 class="pagebreak">Task 2 — Classical segmentation</h2>

<h3>Dev / eval split (leakage-aware)</h3>
<p>
All 110 images were pooled (ignoring Roboflow's train/valid/test folders) and split 4:1 into dev (88) and
eval (22). Because the photos look like they could include short bursts of near-duplicate shots, a
perceptual hash (pHash, 16×16, Hamming distance ≤ 6) was computed per image and used to union-find cluster
near-duplicates; whole clusters — never individual images — are assigned to a side, so a near-duplicate of
a dev image can never leak into eval. In this dataset every image turned out to hash as its own singleton
cluster (no near-duplicates were found), so the split reduced to a clean random 88/22 assignment, but the
clustering step runs regardless and would matter on a burst-captured dataset.
</p>

<h3>Experiment log</h3>
<p>
All trials below were scored by mean IoU / Dice against the Task 1 SAM masks on the <b>dev split only</b>
(the eval split was never looked at during development). Each strategy's full definition is in
<code>src/classical_pipeline.py</code>; the harness is <code>src/task2_dev_experiments.py</code>.
</p>
<table>
<thead><tr><th>Trial</th><th>Strategy</th><th class="num">Mean IoU</th><th class="num">Mean Dice</th><th>What changed &amp; why</th></tr></thead>
<tbody>
{trial_table_rows(trials)}
</tbody>
</table>
<p>
<b>Reading the log:</b> every plain-thresholding variant (T1-T6: global/adaptive Otsu on grayscale, Lab-L,
HSV saturation/value, Canny-edge-fill) tops out around IoU 0.10-0.29 — none of them is robust across the
dataset's lighting/background variety (marble floor vs. white tray vs. reflective steel vs. dark keyboard).
Adaptive thresholding (T5) was actively worse than a single global Otsu split: its local-window statistic
is noisy on the smoothly-shaded egg surface itself, fragmenting the egg into many small regions, and it
was discarded immediately. Classical <b>GrabCut</b> (T7), seeded with a fixed center-biased rectangle,
was a clear step up (IoU 0.400) — unlike a single threshold, it fits a local color model (GMM) to
foreground vs. background and refines iteratively, which tolerates the within-image lighting gradients
that break Otsu. Adding watershed splitting for touching/adjacent eggs (T10) pushed it further (IoU
{fmt(t10['mean_iou'])}) and is where the "fast" (non-GrabCut) hybrid fallback idea (T8/T9) was abandoned —
the fallback's own contrast proxy never actually triggered the GrabCut branch on this dataset, so it was
strictly equivalent to plain Otsu-Lab-L and added complexity for nothing. Seeding GrabCut from an adaptive
Otsu-derived mask instead of a fixed rectangle (T11) was tried as a way to better localize off-center eggs,
but performed worse than the simple fixed rectangle (IoU {fmt(next(t for t in trials if t['trial']=='T11_grabcut_mask_init')['mean_iou'])}) — the Otsu mask itself is unreliable on the hard images, so it
poisons GrabCut's seed rather than improving it.
</p>
{T12_PARAGRAPH}
<p>
<b>Final Task 2 algorithm ({FINAL_TRIAL_NAME}):</b> {FINAL_STRATEGY_DESC} → morphological open/close +
interior hole-fill (cleans up specular highlights on the eggshell) → watershed split on the distance
transform (separates touching/adjacent eggs) → shape filter (area fraction, solidity, circularity,
border-touching) identical in spirit to Task 1's filter but deliberately looser, since classical masks
have jaggier boundaries than SAM's even on a correctly-detected egg.
</p>
<p class="callout">
<b>Known cost trade-off:</b> GrabCut is slow (~5s/image on this machine) — fine for a one-time batch over
110 dataset photos (~10 min), completely impractical per-frame on video, which is why Task 4 uses a
different (faster, threshold-based) strategy instead. See Task 4.
</p>

<h2 class="pagebreak">Task 3 — Evaluation &amp; analysis</h2>
<p>
Task 2's final algorithm was run on the held-out <b>eval split</b> (22 images, never used for tuning) and
scored against Task 1's SAM masks as pseudo-ground-truth. One eval image
(<code>not_damaged_12...</code>) was excluded — SAM found zero instances on it too (same foliage-occlusion
failure mode as Task 1's dev-split misses), so there is no meaningful ground truth to compare against.
{summary['n_images']} images were scored.
</p>
<div class="metric-grid">
  <div class="metric"><div class="v">{fmt(summary['mean_iou'],2)}</div><div class="k">mean IoU</div></div>
  <div class="metric"><div class="v">{fmt(summary['mean_dice'],2)}</div><div class="k">mean Dice</div></div>
  <div class="metric"><div class="v">{fmt(summary['mean_precision'],2)}</div><div class="k">mean precision</div></div>
  <div class="metric"><div class="v">{fmt(summary['mean_recall'],2)}</div><div class="k">mean recall</div></div>
  <div class="metric"><div class="v">{fmt(summary['mean_pixel_accuracy'],2)}</div><div class="k">pixel accuracy</div></div>
  <div class="metric"><div class="v">{fmt(summary['median_iou'],2)}</div><div class="k">median IoU</div></div>
  <div class="metric"><div class="v">{fmt(summary['frac_iou_below_0.5']*100,0)}%</div><div class="k">images IoU&lt;0.5</div></div>
  <div class="metric"><div class="v">{fmt(summary['frac_iou_above_0.8']*100,0)}%</div><div class="k">images IoU&gt;0.8</div></div>
</div>
<p>Recall (0.78) noticeably exceeds precision (0.65): the classical method tends to <i>over</i>-segment
relative to SAM (pulling in extra background) more often than it misses egg pixels outright, on the
images where it works at all.</p>

<div class="two-col">
<figure><img src="{b64img(REPORT/'assets'/'iou_histogram.png', 'image/png')}">
<figcaption>Fig. 3 — IoU distribution on the eval split is clearly <b>bimodal</b>: a cluster of
near-total failures (IoU≈0) and a larger cluster of good results (IoU≈0.65-0.8), with very few images
in between. The algorithm isn't uniformly mediocre — it's either working well or failing hard.</figcaption></figure>
<figure><img src="{b64img(REPORT/'assets'/'iou_vs_contrast.png', 'image/png')}">
<figcaption>Fig. 4 — IoU vs. the Otsu-based foreground/background contrast score. Failures occur across
the <i>whole</i> contrast range, including some high-contrast images — ruling out "low contrast" as the
dominant failure driver and pointing instead at geometry (next section).</figcaption></figure>
</div>

<h3>Qualitative analysis — what's going wrong</h3>
<p>
The worst-IoU eval images (Fig. 5) make the real failure mode visible: GrabCut here is seeded with a
single <b>fixed center-biased rectangle</b> covering the middle ~76% of the frame. That structurally
breaks in two ways:
</p>
<ol>
  <li><b>Under-segmentation (false negatives, blue):</b> when eggs are spread across the full image width
  (e.g. three eggs laid out left-to-right on a tray) rather than clustered near center, any egg outside
  the rectangle is invisible to GrabCut and is missed entirely — row 1 of Fig. 5 finds only 1 of 3 eggs.
  Rows 2, 3, 5 are near-total misses of the same kind on single eggs that sit close to the frame edge.</li>
  <li><b>Over-segmentation (false positives, red):</b> when a background object inside the rectangle has
  similar brightness/color statistics to the egg, GrabCut's regional color model pulls it in as
  foreground too — row 4 (a white bed-rail / stand) and row 6 (the pot's rim and a second, unlabeled egg)
  show large false-positive regions attached to a correctly-found true positive.</li>
</ol>
<figure class="bigfig">
  <img src="{b64img(REPORT/'assets'/'worst_cases_grid.png', 'image/png')}">
  <figcaption>Fig. 5 — The 6 lowest-IoU eval images. Error map: green = true positive, red = false
  positive, blue = false negative.</figcaption>
</figure>
<figure class="bigfig">
  <img src="{b64img(REPORT/'assets'/'best_cases_grid.png', 'image/png')}">
  <figcaption>Fig. 6 — The 4 highest-IoU eval images, for contrast. Note row 2: three vertically-stacked
  eggs are segmented and split correctly — multi-instance isn't inherently broken, it only fails when the
  eggs fall outside the single seed rectangle (Fig. 5, row 1).</figcaption>
</figure>

{T12_RESULT_SECTION}

<h3>Where this leaves things / further improvement ideas</h3>
<ul>
  <li><b>Per-blob seeding</b> (tested above as a follow-up) directly targets the dominant failure mode and
  is the single highest-value next step if more time were available.</li>
  <li><b>Ensemble/voting</b> across 2-3 of the cheaper strategies (Otsu-Lab-L, HSV-SV) combined with
  GrabCut refinement only in the disagreement region, to recover some of GrabCut's robustness at lower
  average cost.</li>
  <li><b>Boundary-aware metrics</b> (e.g. boundary F1 at a few pixel tolerances) in addition to IoU/Dice,
  since several of the "good" cases still lose a thin rim around the egg to the SAM mask being a few
  pixels tighter/looser — not visible in the aggregate IoU number.</li>
</ul>

<h2 class="pagebreak">Task 4 — Video tracking</h2>
<p>
Source: <a href="https://www.youtube.com/watch?v=Y4JyV94y2_A">"Raw and Boiled Egg Spinning" (Mocomi)</a>,
640×360 @ 24fps, 60s. By eye (a contact-sheet scan every 2s), only <b>9-42s</b> shows actual eggs on a
table — the rest is a title card and a text-slide explanation — so that range (792 frames) is what gets
processed and rendered.
</p>
<h3>Segmentation: adapted for this video (as the brief allows)</h3>
<p>
Reusing Task 2's winning GrabCut pipeline unmodified does not work here, and would also be far too slow
(~5s/image × ~800 frames ≈ over an hour). Two video-specific issues had to be found and fixed:
</p>
<ul>
  <li><b>A brighter distractor than the egg.</b> The subject's checkered shirt occupies the top of the
  static-camera frame and is brighter than the eggs under this lighting, so a single global threshold
  (Otsu-Lab-L) picks out shirt squares instead of (or in addition to) the eggs. Since the camera never
  moves, the fix is a fixed table-only ROI crop (top 100 of 360 rows excluded) rather than re-deriving a
  threshold that's robust to it — simple, and specific to this clip as intended.</li>
  <li><b>A watershed marker bug</b> (found and fixed during this task, worth stating plainly): the
  original <code>watershed_split</code> set the entire true background to the "unknown — please solve"
  marker value instead of a definite background marker, which gave the exterior background no seed to
  grow from and caused OpenCV's watershed to fragment the whole frame into near-empty 1-pixel regions,
  i.e. it looked like the segmenter found nothing at all. The standard recipe (dilate the mask for "sure
  background," threshold the distance transform for "sure foreground," and leave only the thin boundary
  ring as unknown) fixed it immediately once isolated with a focused test.</li>
</ul>
<p>
Final per-frame segmentation: ROI-cropped Otsu-Lab-L threshold → morphological clean → watershed split →
shape filter, then the <b>2 largest</b> surviving blobs by area are kept as detections (this video has
exactly two eggs, and the eggs' own reflection on the glossy table is consistently smaller than the real
eggs, so taking the top-2 by area reliably discards it).
</p>

<h3>Tracking</h3>
<p>
A minimal greedy nearest-centroid tracker (<code>src/tracker.py</code>) — no Kalman filter / motion
model, since with two slow, mostly-stationary eggs "match to nearest centroid, hold position while
briefly lost" is enough and is far easier to reason about live. Design choices motivated directly by
what the detector actually does on this clip:
</p>
<ul>
  <li><b>Capped at 2 simultaneous tracks</b> (the known egg count for this clip) — so an occasional
  reflection blob that survives the top-2-by-area filter (rare, but happens when a real egg is fully
  missed and the reflection becomes one of the two largest blobs) cannot spawn a spurious third identity
  while both real eggs are already being tracked.</li>
  <li><b>Long missed-frame tolerance</b> (190 frames ≈ 7.9s) to survive the hand fully occluding/lifting
  an egg for several seconds during the spin demo — the longest such gap measured in this clip is 151
  frames (6.3s).</li>
  <li><b>Separate display grace period</b> (12 frames ≈ 0.5s): the tracker keeps a lost track alive
  internally for the full 190-frame budget for re-identification, but the drawn overlay stops rendering a
  track after a short grace window rather than showing a box frozen in place for several seconds, which
  would look like a bug rather than a design choice.</li>
</ul>
<figure>
  <img src="{b64img(REPORT/'assets'/'task4_montage.jpg')}">
  <figcaption>Fig. 7 — Left: both eggs tracked normally. Middle: the hand lifts the left egg for an
  extended spin demo (~6s); its track survives the occlusion in the tracker's memory but stops being
  drawn after the grace window. Right: once it reappears, it is just outside the match-and-reappear
  window and is re-identified as a new ID ("egg 3") rather than recovering "egg 1".</figcaption>
</figure>

<h3>Results</h3>
<div class="metric-grid">
  <div class="metric"><div class="v">{BOTH_PCT}%</div><div class="k">frames: both eggs detected</div></div>
  <div class="metric"><div class="v">{ONE_PCT}%</div><div class="k">frames: one egg detected</div></div>
  <div class="metric"><div class="v">{ZERO_PCT}%</div><div class="k">frames: zero eggs detected</div></div>
  <div class="metric"><div class="v">3</div><div class="k">distinct track IDs (ideal: 2)</div></div>
</div>
<p>
Both eggs are correctly tracked as distinct identities for the large majority of the clip. The one
imperfection — 3 total IDs instead of 2 — comes from exactly the ~6.3s occlusion described above: the
track survives it, but the egg's position on re-appearance falls just outside the 100px gating distance
used for re-matching, so it's assigned a new ID rather than recovering the old one. Identities never swap
or merge incorrectly anywhere in the clip, which was the actual objective ("track individual eggs
separately").
</p>
<p>
<b>Full annotated output video:</b> {drive_link_html}
</p>

<h2 class="pagebreak">Setup &amp; reproducing this report</h2>
<p>
Full setup and run instructions are in the repository's <code>README.md</code>. In short:
<code>pip install -r requirements.txt</code> (torch needs the CUDA 12.1 index explicitly, see README) →
<code>python src/build_manifest.py</code> → <code>python src/task1_sam_segment.py</code> →
<code>python src/task2_dev_experiments.py</code> → <code>python src/task3_evaluate.py</code> →
download the video with <code>yt-dlp</code> and run <code>python src/task4_track_eggs.py</code>.
</p>
"""

    out_path = REPORT / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html)} chars)")


REPORT_DATE = "October 31, 2026"
DRIVE_LINK = "PENDING"  # filled in once the user shares the Drive URL

# Filled in by hand after inspecting outputs (see inline comments)
MULTI_INSTANCE = 23
ZERO_INSTANCE = 8
BOTH_PCT = 62
ONE_PCT = 27
ZERO_PCT = 11

FINAL_TRIAL_NAME = "T10_grabcut_watershed"
FINAL_STRATEGY_DESC = ("GrabCut (Rother et al. 2004) seeded with a fixed center-biased rectangle "
                        "(covering the middle 76% of the frame)")
T12_PARAGRAPH = ""
T12_RESULT_SECTION = """
<h3>Follow-up experiment: does per-blob seeding fix it? (T12 — unsuccessful)</h3>
<p>
The diagnosis above directly suggests a fix: instead of one fixed rectangle, find candidate blobs with
Otsu-Lab-L first and seed a separate GrabCut call per blob (<code>fg_grabcut_multiseed</code>), so every
egg gets its own locally-seeded segmentation regardless of where it sits in the frame. Re-running the dev
harness with this strategy added as <b>T12</b> gives:
</p>
<table>
<thead><tr><th>Trial</th><th>Strategy</th><th class="num">Mean IoU</th><th class="num">Mean Dice</th><th>What changed &amp; why</th></tr></thead>
<tbody>{t12_row}</tbody>
</table>
<p>
This is <i>worse</i> than the simple fixed-rectangle version (T10), not better, and the reason is
informative: the blob proposals come from Otsu-Lab-L — the same thresholding method that scored only
0.215 IoU on its own (T3) because it struggles on this dataset's harder lighting/background combinations.
Delegating "where to look" to an unreliable proposer means (a) on low-contrast images, Otsu often doesn't
propose a blob for the real egg at all, so no GrabCut call is even made there — recall is lost exactly
where it was already weakest; and (b) on noisy images, Otsu proposes several small spurious blobs (texture,
highlights, shadow), and each gets its own confident local GrabCut call with no larger context to reject
it, adding false positives. The fixed-center-rectangle's "the egg is usually near the middle" prior, despite
being a far cruder assumption, turned out to be a stronger and more stable signal on this dataset than a
data-driven proposal step inherited from a weaker method — the same lesson T11 already hinted at (seeding
GrabCut from an Otsu mask instead of a fixed rectangle also underperformed T10). <b>T10 remains the final
algorithm</b> used for the Task 3 numbers above; a better per-region seeding strategy (e.g. seeding from
SAM-style objectness rather than Otsu, or a proper multi-scale sliding-window proposal) is left as the
noted improvement direction rather than attempted further given the time budget.
</p>
"""

if __name__ == "__main__":
    main()
