"""
Task 2 - Classical (non-learned) egg segmentation pipeline.

This module implements several *interchangeable* foreground-extraction
strategies plus shared post-processing (morphological cleanup, shape-based
component filtering, and watershed splitting for touching eggs). The
experiment harness (dev_experiments.py) swaps strategies in and out and scores
them against the dev split to decide what to keep; this file is the end state
after that process, but every strategy function is kept (not deleted) because
the harness and the report's experiment log re-run them for comparison.
"""
import cv2
import numpy as np

from common import mask_shape_features


# ---------------------------------------------------------------------------
# Foreground-extraction strategies (each returns a uint8 0/255 binary mask)
# ---------------------------------------------------------------------------

def fg_otsu_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can pick either side as "foreground"; assume the egg is the
    # minority-area, more-central class relative to a full image Otsu split.
    if mask.mean() > 127:
        mask = 255 - mask
    return mask


def fg_otsu_lab_l(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    _, mask = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:
        mask = 255 - mask
    return mask


def fg_hsv_sv(img_bgr):
    """Low-saturation AND high-value pixels (targets white/light eggshell)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    s_thresh, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thresh, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((s <= s_thresh) & (v >= v_thresh)).astype(np.uint8) * 255
    return mask


def fg_adaptive_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
    )
    return mask


def fg_canny_fill(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def fg_grabcut(img_bgr, rect_frac=(0.12, 0.12, 0.88, 0.88), iters=5):
    """Classical iterative graph-cut segmentation (Rother et al. 2004),
    seeded with a fixed center-biased rectangle (the dataset consistently
    frames the egg away from the extreme borders, see manifest inspection)."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = rect_frac
    rect = (int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg


def fg_grabcut_mask_init(img_bgr, iters=5, border_frac=0.04):
    """GrabCut seeded from an Otsu-Lab-L foreground estimate (as "probable
    foreground") instead of a fixed rectangle, so it adapts to eggs that are
    off-center or a different size than the dataset's typical framing. A thin
    border ring is seeded as definite background (the dataset/video never
    crops the egg at the very edge)."""
    h, w = img_bgr.shape[:2]
    probable_fg = fg_otsu_lab_l(img_bgr) > 0

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_fg] = cv2.GC_PR_FGD
    bw = max(2, int(border_frac * min(h, w)))
    gc_mask[:bw, :] = cv2.GC_BGD
    gc_mask[-bw:, :] = cv2.GC_BGD
    gc_mask[:, :bw] = cv2.GC_BGD
    gc_mask[:, -bw:] = cv2.GC_BGD

    if not probable_fg.any():
        return np.zeros((h, w), dtype=np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg


def fg_grabcut_multiseed(img_bgr, iters=5, pad_frac=0.25, min_blob_area_frac=0.005):
    """GrabCut seeded per-blob instead of a single fixed center rectangle.

    Diagnosis (see Task 3 qualitative analysis): the single-rectangle GrabCut
    (fg_grabcut) only sees the fixed center 76% of the frame, so when eggs
    are spread across the full width (e.g. 3 eggs laid out left-to-right) any
    egg outside that rectangle is missed entirely (FN), and conversely a
    background object that happens to fall inside the rectangle can get
    pulled in as a false positive. Using Otsu-Lab-L connected components to
    propose one rectangle per candidate blob, then running GrabCut separately
    in each, lets every blob get its own locally-seeded segmentation.
    """
    h, w = img_bgr.shape[:2]
    probable_fg = fg_otsu_lab_l(img_bgr)
    probable_fg = cv2.morphologyEx(
        probable_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(probable_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if cv2.contourArea(c) >= min_blob_area_frac * h * w]

    if not blobs:
        return fg_grabcut(img_bgr)  # fall back to the single center-rect seed

    combined = np.zeros((h, w), dtype=np.uint8)
    for c in blobs:
        x, y, bw, bh = cv2.boundingRect(c)
        px, py = int(bw * pad_frac), int(bh * pad_frac)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
        rect = (x0, y0, x1 - x0, y1 - y0)

        gc_mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            continue
        fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        combined[fg] = 255
    return combined


def contrast_score(img_bgr):
    """Cheap proxy for foreground/background separability: Otsu's
    between-class variance on the Lab L channel, normalized. Low score ~ low
    contrast (e.g. white egg on a white tray)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    thresh, _ = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = l_chan[l_chan >= thresh]
    bg = l_chan[l_chan < thresh]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return float(abs(fg.mean() - bg.mean()))


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def morphological_clean(mask, open_k=7, close_k=15):
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    # fill small interior holes (specular highlights on the eggshell)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def watershed_split(mask, peak_frac=0.4):
    """Separate touching/overlapping blobs (e.g. two eggs side by side in a
    pot) using the standard distance-transform + marker-based watershed
    recipe. Returns a list of per-instance binary masks.

    Marker setup (this is the part that's easy to get backwards): sure_fg is
    the small core near each blob's distance-transform peak (one marker label
    per object); sure_bg is the true background, obtained by dilating the
    mask slightly -- everything *outside* that dilation is unambiguously
    background and keeps a single shared marker label; only the thin ring
    between sure_fg and sure_bg ("unknown") is left as 0 for watershed to
    resolve. Setting the true exterior background itself to 0 (unknown) is a
    bug: it then has no seed to grow from and watershed can fragment it into
    noise across the whole frame.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    _, sure_fg = cv2.threshold(dist, peak_frac * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:  # 0 or 1 real foreground component -> nothing to split
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            m = np.zeros_like(mask)
            cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
            out.append(m > 0)
        return out

    markers = markers + 1
    markers[unknown == 255] = 0
    img3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    instances = []
    for label in range(2, n_markers + 1):
        inst = (markers == label)
        if inst.sum() > 0:
            instances.append(inst)
    return instances


def mean_gradient_magnitude(gray, mask):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[mask]
    return float(vals.mean()) if vals.size else 0.0


def filter_egg_like_components(instance_masks, min_area_frac=0.012, min_solidity=0.80,
                                 min_circularity=0.45, max_touches_borders=2,
                                 gray=None, max_texture=None):
    """Looser than the SAM post-filter (Task 1) on purpose: classical masks
    are noisier (jagged boundaries lower circularity/solidity even for true
    eggs), so overly strict thresholds here would reject correct detections
    rather than reject false positives. Tuned on the dev split only.

    gray/max_texture: optional -- an eggshell is smooth (low internal gradient
    magnitude), whereas a patterned object of similar brightness and
    convexity after morphological closing (e.g. a checkered shirt square in
    the Task 4 video) is not. Only used when both are provided."""
    kept = []
    for m in instance_masks:
        feat = mask_shape_features(m)
        if feat is None:
            continue
        if feat["touches_borders"] > max_touches_borders:
            continue
        if not (min_area_frac <= feat["area_frac"] <= 0.7):
            continue
        if feat["solidity"] < min_solidity:
            continue
        if feat["circularity"] < min_circularity:
            continue
        if gray is not None and max_texture is not None:
            if mean_gradient_magnitude(gray, m) > max_texture:
                continue
        kept.append(m)
    return kept


# ---------------------------------------------------------------------------
# Full pipelines (strategy + fixed post-processing), as used by the harness
# ---------------------------------------------------------------------------

STRATEGIES = {
    "otsu_gray": fg_otsu_gray,
    "otsu_lab_l": fg_otsu_lab_l,
    "hsv_sv": fg_hsv_sv,
    "adaptive_gray": fg_adaptive_gray,
    "canny_fill": fg_canny_fill,
    "grabcut": fg_grabcut,
    "grabcut_mask_init": fg_grabcut_mask_init,
    "grabcut_multiseed": fg_grabcut_multiseed,
}


def segment(img_bgr, strategy="otsu_lab_l", clean=True, split_touching=True,
            filter_components=True, grabcut_fallback_thresh=None,
            roi_y0=0, use_texture_filter=False, max_texture=70.0):
    """Returns (combined_mask uint8 0/255, list of per-instance bool masks).

    roi_y0: crop out the top `roi_y0` rows before segmenting (Task 4: the
    video's camera is static with a bright patterned shirt occupying the top
    of the frame that otherwise dominates a global threshold; cropping it out
    is simpler and far cheaper than re-deriving a strategy that's robust to
    it). use_texture_filter: additionally reject components with high
    internal gradient magnitude (Task 4: tells a smooth eggshell apart from a
    checkered-fabric square of similar brightness/convexity).
    """
    full_h, full_w = img_bgr.shape[:2]
    work_img = img_bgr[roi_y0:, :] if roi_y0 > 0 else img_bgr

    fg_fn = STRATEGIES[strategy]
    mask = fg_fn(work_img)

    if grabcut_fallback_thresh is not None and strategy != "grabcut":
        if contrast_score(work_img) < grabcut_fallback_thresh:
            mask = fg_grabcut(work_img)

    if clean:
        mask = morphological_clean(mask)

    if split_touching:
        instances = watershed_split(mask)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        instances = []
        for c in contours:
            m = np.zeros_like(mask, dtype=bool)
            cv2.drawContours(m.view(np.uint8), [c], -1, 1, thickness=cv2.FILLED)
            instances.append(m)

    if filter_components:
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY) if use_texture_filter else None
        instances = filter_egg_like_components(
            instances, gray=gray, max_texture=max_texture if use_texture_filter else None)

    if roi_y0 > 0:
        instances = [np.pad(m, ((roi_y0, 0), (0, 0))) for m in instances]

    combined = np.zeros((full_h, full_w), dtype=np.uint8)
    for m in instances:
        combined[m] = 255
    return combined, instances
