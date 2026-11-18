# make_students.py
# Generate 4 student masks (s1..s4) with controlled differences to GT.
# Dependencies: opencv-python, numpy
# Usage:
#   python make_students.py                 # uses expected/lesion_mask-reference.png
#   python make_students.py --gt path/to/ISIC_0000000_segmentation.png

import os, argparse
import numpy as np
import cv2

# --------------------------- config you can tweak -----------------------------

# Targets for the *full-mask* IoU. The tiny search below will pick erosion/dilation
# iterations to land near these numbers on your image.
TARGET_IOU_S2 = 0.986   # under-segmentation target (lower == stronger erosion)
TARGET_IOU_S3 = 0.977   # over-segmentation  target (lower == stronger dilation)

# S4: make it obviously wrong by shifting + eroding.
S4_SHIFT = (9, -6)      # (dx, dy) in pixels
S4_ERODE_ITERS = 3

# ROI ring width ≈ 6 px achieved by 3 iterations with a 3x3 structuring element.
ROI_ITERS = 3

# Default GT path (adjust if needed, or pass --gt)
DEFAULT_GT = os.path.join("expected", "lesion_mask-reference.png")

# -----------------------------------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_bin(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT mask not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # binarize to {0,255}
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw

def write_bin(path, mask):
    mask = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(path, mask)

def iou(a, b):
    A = a > 0
    B = b > 0
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / max(1, union)

def roi_ring(gt, iters=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dil = cv2.dilate(gt, k, iterations=iters)
    ero = cv2.erode (gt, k, iterations=iters)
    ring = cv2.subtract(dil, ero)
    return (ring > 0)

def n_diff_in_ring(gt, stud, ring_bool):
    diff = np.logical_xor(gt>0, stud>0)
    return int(np.logical_and(diff, ring_bool).sum())

def morph_to_target(gt, mode="erode", target=0.985, iters_max=4):
    """
    Move the mask slightly toward/away from boundary until IoU ≈ target.
    Returns (cand, achieved_iou, used_iters)
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    best = gt.copy()
    best_iou = iou(gt, gt)
    for it in range(1, iters_max+1):
        if mode == "erode":
            cand = cv2.erode(gt, k, iterations=it)
        else:
            cand = cv2.dilate(gt, k, iterations=it)
        cur = iou(gt, cand)
        # choose the first that drops to (or just past) target
        if cur <= target:
            return cand, cur, it
        best, best_iou = cand, cur
    return best, best_iou, iters_max

def translate(mask, dx, dy):
    h, w = mask.shape[:2]
    M = np.float32([[1,0,dx],[0,1,dy]])
    shifted = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return shifted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, default=DEFAULT_GT,
                    help="Path to GT binary mask (white=lesion, black=bg).")
    args = ap.parse_args()

    gt = read_bin(args.gt)

    # Prepare output folders
    root = os.path.abspath(os.path.join(os.path.dirname(args.gt), ".."))
    students_root = os.path.join(root, "students")
    s1_dir = os.path.join(students_root, "S1")
    s2_dir = os.path.join(students_root, "S2")
    s3_dir = os.path.join(students_root, "S3")
    s4_dir = os.path.join(students_root, "S4")
    for d in (s1_dir, s2_dir, s3_dir, s4_dir):
        ensure_dir(d)

    # ROI ring for reporting N_DIFF
    ring_bool = roi_ring(gt, ROI_ITERS)

    outname = "lesion_mask_stud.png"

    # ------------------------ S1: near-perfect -------------------------
    # Optionally smooth a hair to remove single-pixel jaggies, then re-binarize.
    s1 = gt.copy()
    # If you want literally GT, comment the next two lines.
    s1 = cv2.GaussianBlur(s1, (3,3), 0.8)
    _, s1 = cv2.threshold(s1, 127, 255, cv2.THRESH_BINARY)

    # ------------------------ S2: under-seg ----------------------------
    s2, s2_iou, s2_it = morph_to_target(gt, "erode", TARGET_IOU_S2, iters_max=4)

    # ------------------------ S3: over-seg -----------------------------
    s3, s3_iou, s3_it = morph_to_target(gt, "dilate", TARGET_IOU_S3, iters_max=4)

    # ------------------------ S4: clearly wrong ------------------------
    s4 = translate(gt, S4_SHIFT[0], S4_SHIFT[1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    s4 = cv2.erode(s4, k, iterations=S4_ERODE_ITERS)

    # ------------------------ Save + report ----------------------------
    sets = [
        ("S1", s1),
        ("S2", s2),
        ("S3", s3),
        ("S4", s4),
    ]

    print("Done. Wrote:")
    for sid, m in sets:
        path = os.path.join(students_root, sid, outname)
        write_bin(path, m)
        print(f" - {path}")

    # metrics printout for your sanity check
    print("\nSummary versus GT")
    print("ID  |  IoU      |  N_DIFF(ROI)")
    print("--------------------------------")
    for sid, m in sets:
        the_iou = iou(gt, m)
        ndiff = n_diff_in_ring(gt, m, ring_bool)
        print(f"{sid:>2}  |  {the_iou:0.6f}  |  {ndiff}")

    # extra details for S2/S3 so you know what was chosen
    print("\nDetails:")
    print(f"S2 (erode): target {TARGET_IOU_S2}, used {s2_it} iters, achieved IoU={iou(gt,s2):0.6f}")
    print(f"S3 (dilate): target {TARGET_IOU_S3}, used {s3_it} iters, achieved IoU={iou(gt,s3):0.6f}")

if __name__ == "__main__":
    main()
