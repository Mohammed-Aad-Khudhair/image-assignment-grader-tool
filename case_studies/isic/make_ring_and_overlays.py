# make_ring_and_overlays.py
# Produce figure overlays for GT and S1..S4 with a visible ROI ring and
# ring-difference maps (cyan=miss, magenta=overflow).
#
# Run after make_students.py has created students/S1..S4/lesion_mask_stud.png

import os
from pathlib import Path
import cv2
import numpy as np

BASE     = Path(__file__).resolve().parent
EXP_DIR  = BASE / "expected"
STUD_DIR = BASE / "students"
FIG_DIR  = BASE / "figures"
MASK_DIR = BASE / "masks"

FIG_DIR.mkdir(exist_ok=True, parents=True)
MASK_DIR.mkdir(exist_ok=True, parents=True)

# ---- visualization params ----
RING_ITERS      = 4          # widen ROI ring (dilate/erode iterations) ~8 px
RING_THICK      = 6          # thickness of ring border strokes
RING_FILL_ALPHA = 0.35       # gray fill alpha inside the ROI band
BORDER_THICK    = 5          # thickness for GT / student boundaries

# BGR colors (OpenCV)
GREEN   = (  0,255,  0)      # GT boundary
ORANGE  = ( 40,170,255)      # student boundary
YELLOW  = ( 30,230,255)      # outer ROI border
WHITE   = (255,255,255)      # inner ROI border
CYAN    = (255,255,  0)      # miss
MAGENTA = (255,  0,255)      # overflow

# ---------- helpers ----------
def find_gt_mask():
    candidates = [
        "lesion_mask.png",
        "lesion_mask-reference.png",
        "ISIC_0000000_segmentation.png",
    ]
    for name in candidates:
        p = EXP_DIR / name
        if p.exists():
            return p
    for p in sorted(EXP_DIR.glob("*.png")):
        if "mask" in p.name.lower() or "segment" in p.name.lower():
            return p
    raise FileNotFoundError(
        f"No ground-truth mask found in {EXP_DIR}. "
        "Expected e.g. lesion_mask.png or *segmentation*.png"
    )

def find_bg_image():
    # prefer lesion.jpg; else any jpg in expected; else return None
    preferred = ["lesion.jpg", "Derm.jpg", "ISIC_0000000.jpg"]
    for name in preferred:
        p = EXP_DIR / name
        if p.exists():
            return p
    imgs = list(EXP_DIR.glob("*.jpg"))
    return imgs[0] if imgs else None

def ensure_binary(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return b

def read_bin(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return ensure_binary(m)

def read_color_fallback(path: Path, size_hw):
    if path is None or (not path.exists()):
        H, W = size_hw
        return np.full((H, W, 3), 32, np.uint8)  # dark gray fallback
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        H, W = size_hw
        return np.full((H, W, 3), 32, np.uint8)
    H, W = size_hw
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img

def mask_edges(mask, thickness=3):
    """Return a binary edge map by drawing the contour."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edges = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(edges, cnts, -1, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return edges

def paint_color(img_bgr, mask, color_bgr, alpha=0.6):
    """Alpha-paint 'color_bgr' wherever 'mask>0'."""
    out = img_bgr.copy()
    idx = mask > 0
    if idx.any():
        overlay = out.copy()
        overlay[idx] = color_bgr
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out

def bbox_from_mask(mask, pad=40):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    h, w = mask.shape
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
    return (x0, y0, x1, y1)

# ---------- build ROI ring + edges ----------
GT_PATH = find_gt_mask()
gt     = read_bin(GT_PATH)

kernel = np.ones((3,3), np.uint8)
outer  = cv2.dilate(gt, kernel, iterations=RING_ITERS)
inner  = cv2.erode (gt, kernel, iterations=RING_ITERS)
ring   = cv2.subtract(outer, inner)
cv2.imwrite(str(MASK_DIR / "roi_border.png"), ring)

edge_gt         = mask_edges(gt,    thickness=BORDER_THICK)
edge_ring_outer = mask_edges(outer, thickness=RING_THICK)
edge_ring_inner = mask_edges(inner, thickness=RING_THICK)

H, W = gt.shape
IMG_BG = find_bg_image()
bg     = read_color_fallback(IMG_BG, (H, W))

# ---------- overlay generators ----------
def overlay_gt(out_path: Path):
    out = bg.copy()
    out[edge_gt > 0] = GREEN
    cv2.imwrite(str(out_path), out)
    print(f"wrote: {out_path}")

def overlay_boundaries(stud_mask, out_path: Path):
    out = bg.copy()
    # draw GT first (green), then student (orange) so mismatches stand out
    out[edge_gt > 0] = GREEN
    edge_stud = mask_edges(stud_mask, thickness=BORDER_THICK)
    out[edge_stud > 0] = ORANGE
    cv2.imwrite(str(out_path), out)
    print(f"wrote: {out_path}")

def overlay_ring_diff(stud_mask, out_path: Path, zoom_path: Path):
    out = bg.copy()

    # gray fill for ROI band
    out = paint_color(out, ring, (180,180,180), alpha=RING_FILL_ALPHA)

    # two distinct ring borders
    out[edge_ring_outer > 0] = YELLOW
    out[edge_ring_inner > 0] = WHITE

    # restrict diffs to the ring
    miss = ((gt == 255) & (stud_mask == 0)   & (ring == 255)).astype(np.uint8)*255
    over = ((gt == 0)   & (stud_mask == 255) & (ring == 255)).astype(np.uint8)*255

    out = paint_color(out, miss, CYAN,    alpha=0.75)
    out = paint_color(out, over, MAGENTA, alpha=0.75)

    cv2.imwrite(str(out_path), out)
    print(f"wrote: {out_path}")

    # zoom if there are any ring differences
    diff = cv2.bitwise_or(miss, over)
    box = bbox_from_mask(diff, pad=48)
    if box:
        x0, y0, x1, y1 = box
        crop = out[y0:y1+1, x0:x1+1]
        cv2.imwrite(str(zoom_path), crop)
        print(f"wrote: {zoom_path}")
    else:
        print(f"[{out_path.stem.split('_')[1]}] no ring differences; skipping zoom.")

# ---------- main ----------
def main():
    # write the GT-only overlay once
    overlay_gt(FIG_DIR / "overlay_GT.png")

    # per-student overlays
    for sid in ["S1", "S2", "S3", "S4"]:
        stud_path = STUD_DIR / sid / "lesion_mask_stud.png"
        if not stud_path.exists():
            print(f"[WARN] missing: {stud_path} (skipping)")
            continue
        stud = read_bin(stud_path)

        overlay_boundaries(stud, FIG_DIR / f"overlay_{sid.upper()}_boundaries.png")
        overlay_ring_diff(stud,
                          FIG_DIR / f"overlay_{sid.upper()}_diff_ring.png",
                          FIG_DIR / f"zoom_{sid.upper()}.png")

if __name__ == "__main__":
    main()
