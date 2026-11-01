# Egg Segmentation & Tracking — CV Intern Task

Four-part computer vision pipeline on the [Roboflow egg dataset](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1):

1. **Task 1** — segment eggs in every dataset image with a pretrained [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).
2. **Task 2** — segment eggs with classical (non-learned) image processing, developed on an 80/20 dev/eval split.
3. **Task 3** — evaluate Task 2 against Task 1 (treated as pseudo-ground-truth) on the held-out eval split.
4. **Task 4** — track the two eggs individually through a [YouTube demo video](https://www.youtube.com/watch?v=Y4JyV94y2_A).

The full write-up (methodology, experiment log, results, analysis, video link) is in [`report/CV_Intern_Report.pdf`](report/CV_Intern_Report.pdf).

## Setup

Requires Python 3.11 and an NVIDIA GPU (CUDA) for a reasonable Task 1 runtime (SAM also runs on CPU, just much slower).

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate

# Torch is pinned to a CUDA 12.1 build; install it from PyTorch's index first
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

Download the dataset (Roboflow "folder" export) and unzip into `data/raw/`, so you have:

```
data/raw/{train,valid,test}/egg/*.jpg
```

Download the SAM ViT-B checkpoint (375MB; ViT-B was chosen over ViT-H/ViT-L to fit a 4GB-VRAM GPU — see report):

```bash
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Running the pipeline

All scripts are run from the repo root with the venv active, `src/` added implicitly via each script's own path handling (run as `python src/<script>.py`, not from inside `src/`).

```bash
# 1. Build the image manifest + dev/eval split (perceptual-hash de-duplication
#    guards against near-identical frames leaking across the split)
python src/build_manifest.py

# 2. Task 1: SAM segmentation on all 110 images (~2-3 min on a GTX 1650)
python src/task1_sam_segment.py

# 3. Task 2: run the experiment log (9 strategies x dev split), writes
#    report/experiment_log_results.csv. GrabCut trials are slow (~5s/image).
python src/task2_dev_experiments.py

# 4. Task 3: evaluate the final Task 2 config against Task 1 on the eval
#    split; writes report/eval_metrics_per_image.csv, eval_summary.json,
#    and qualitative figures under report/assets/
python src/task3_evaluate.py

# 5. Task 4: download the source video, then track the eggs
yt-dlp --extractor-args "youtube:player_client=android" \
  -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best" \
  -o "data/video/source.%(ext)s" "https://www.youtube.com/watch?v=Y4JyV94y2_A"
python src/task4_track_eggs.py
# outputs/task4_tracking/tracked.mp4 is written with OpenCV's mp4v codec;
# re-encode with ffmpeg for reliable browser/Drive playback:
python -c "import imageio_ffmpeg,subprocess; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i','outputs/task4_tracking/tracked.mp4','-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-movflags','+faststart','outputs/task4_tracking/tracked_h264.mp4'])"
```

## Repo layout

```
src/
  build_manifest.py        dataset pooling + leakage-aware dev/eval split
  task1_sam_segment.py     Task 1: SAM + heuristic egg-mask selection
  common.py                shared mask-geometry helpers (Task 1 & 2)
  classical_pipeline.py    Task 2: classical segmentation strategies + post-processing
  task2_dev_experiments.py Task 2: experiment harness (dev-split scoring)
  task3_evaluate.py        Task 3: eval-split metrics + qualitative figures
  tracker.py                Task 4: lightweight centroid tracker
  task4_track_eggs.py      Task 4: video segmentation + tracking driver
data/
  raw/                     (gitignored) unzipped Roboflow export
  manifest.csv, splits/    generated dev/eval split
  video/                   (gitignored) downloaded source video
checkpoints/               (gitignored) SAM weights
outputs/                   (gitignored) generated masks/overlays/video
report/
  CV_Intern_Report.pdf     full report
  experiment_log_results.csv, eval_metrics_per_image.csv, eval_summary.json
  assets/                  figures used in the report
```

## AI tool disclosure

This repo (code, experiment design, and report) was built with Claude Code (Anthropic) as a pair-programming assistant. All algorithmic decisions — SAM post-filtering heuristics, the classical segmentation trials, the dev/eval split design, the tracker design — were reviewed and are understood well enough to walk through and modify live, as the assignment expects.
