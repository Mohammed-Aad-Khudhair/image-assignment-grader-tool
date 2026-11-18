import os
import glob
import cv2
import numpy as np
import argparse
from skimage.morphology import skeletonize


def binarize(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img > 0).astype(np.uint8)


def save_mask(path: str, mask_uint1: np.ndarray) -> None:
    out = (mask_uint1.astype(np.uint8) * 255)
    cv2.imwrite(path, out)


def make_cl_ring(gt_bin: np.ndarray, fov_bin: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Centerline ring:
    1) skeletonize GT to 1px lines
    2) dilate by 'radius' to form a thin band around centerlines
    3) clamp to FOV
    """
    skel = skeletonize(gt_bin.astype(bool)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.dilate(skel, k)
    ring = (ring & fov_bin).astype(np.uint8)
    return ring


def make_bg_mask(
    gt_bin: np.ndarray,
    fov_bin: np.ndarray,
    mode: str = "ring",
    inner_radius: int = 3,
    outer_radius: int = 7,
) -> np.ndarray:
    """
    Background mask options:
      - mode='ring': a halo between inner_radius and outer_radius away from vessels
      - mode='full': entire background inside FOV (FOV minus vessels)
    """
    if mode == "full":
        bg = (fov_bin & (~gt_bin))  # entire background inside FOV
        return bg.astype(np.uint8)

    # 'ring' (default): between two dilations, exclude actual vessels
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_radius + 1, 2 * inner_radius + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_radius + 1, 2 * outer_radius + 1))

    d_in = cv2.dilate(gt_bin, k_in)     # near vessels (to exclude)
    d_out = cv2.dilate(gt_bin, k_out)   # slightly farther (to include)
    ring = ((d_out & fov_bin) & (~d_in)) & (~gt_bin)
    return (ring > 0).astype(np.uint8)


def process_one(
    expected_dir: str,
    img_id: str,
    skel_dilate_radius: int,
    bg_mode: str,
    bg_inner_radius: int,
    bg_outer_radius: int,
) -> None:
    gt_path = os.path.join(expected_dir, f"{img_id}_vessels_gt.png")
    fov_path = os.path.join(expected_dir, f"{img_id}_fov.png")
    cl_out = os.path.join(expected_dir, f"{img_id}_centerline_ring.png")
    bg_out = os.path.join(expected_dir, f"{img_id}_bg_ring.png")

    gt_bin = binarize(gt_path)
    if os.path.exists(fov_path):
        fov_bin = binarize(fov_path)
    else:
        # Fallback: if no FOV provided, assume whole image is valid
        fov_bin = np.ones_like(gt_bin, dtype=np.uint8)

    if gt_bin.shape != fov_bin.shape:
        raise ValueError(f"Shape mismatch {gt_bin.shape} vs {fov_bin.shape} for {img_id}")

    cl_ring = make_cl_ring(gt_bin, fov_bin, skel_dilate_radius)
    bg_mask = make_bg_mask(gt_bin, fov_bin, bg_mode, bg_inner_radius, bg_outer_radius)

    save_mask(cl_out, cl_ring)
    save_mask(bg_out, bg_mask)

    # small console summary
    cl_n = int(cl_ring.sum())
    bg_n = int(bg_mask.sum())
    print(f"[{img_id}] saved: {os.path.basename(cl_out)} (px={cl_n}), "
          f"{os.path.basename(bg_out)} (px={bg_n})")


def main():
    ap = argparse.ArgumentParser(
        description="Build per-image CL and BG masks for HRF-style vessel GT and FOV."
    )
    ap.add_argument("--expected_dir", default="./expected", help="Folder with *_vessels_gt.png and *_fov.png")
    ap.add_argument(
        "--ids",
        nargs="*",
        help="Optional list of image IDs (e.g., H01 G12 DR07). If omitted, auto-discovers all *_vessels_gt.png."
    )
    ap.add_argument("--skel_dilate", type=int, default=2, help="Half-thickness of CL ring (pixels)")
    ap.add_argument("--bg_mode", choices=["ring", "full"], default="ring",
                    help="Background mask type: 'ring' (halo) or 'full' (FOV minus vessels)")
    ap.add_argument("--bg_inner", type=int, default=3, help="Inner exclusion radius for BG ring")
    ap.add_argument("--bg_outer", type=int, default=7, help="Outer inclusion radius for BG ring")
    args = ap.parse_args()

    expected_dir = args.expected_dir

    if args.ids:
        ids = args.ids
    else:
        # Auto-discover IDs from *_vessels_gt.png
        patt = os.path.join(expected_dir, "*_vessels_gt.png")
        ids = [os.path.basename(p).replace("_vessels_gt.png", "") for p in glob.glob(patt)]
        ids.sort()

    if not ids:
        raise SystemExit("No *_vessels_gt.png found (and no --ids provided).")

    print(f"Expected dir: {expected_dir}")
    print(f"Images: {', '.join(ids)}")
    print(f"CL ring radius: {args.skel_dilate} | BG mode: {args.bg_mode} "
          f"(inner={args.bg_inner}, outer={args.bg_outer})")

    for img_id in ids:
        process_one(
            expected_dir=expected_dir,
            img_id=img_id,
            skel_dilate_radius=args.skel_dilate,
            bg_mode=args.bg_mode,
            bg_inner_radius=args.bg_inner,
            bg_outer_radius=args.bg_outer,
        )


if __name__ == "__main__":
    main()
