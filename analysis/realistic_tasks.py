"""Six typical course tasks with several correct solutions and typical mistakes.

For each task the instructor's reference is computed with OpenCV. "Correct"
submissions solve the same task with other libraries or by hand (Pillow,
scikit-image, SciPy, NumPy); "wrong" submissions contain a typical mistake.
All submissions are graded with ImgGrader's batch evaluation twice:

  initial     first-guess tolerances an instructor might choose without testing;
  calibrated  tolerances set so that the best band still contains the worst
              correct solution (and, for Sobel, a mask that ignores the 3-pixel
              image border, where libraries treat borders differently).

Calibration uses only the correct solutions, never the mistakes.

Images: the ISIC lesion photo (case_studies/isic) and the scikit-image "coffee"
photo (test_kit/generator/sources, CC0).

Output: results/realistic_tasks.csv (one row per rubric, task and submission)
"""
import os
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from skimage import color, filters, morphology

from _imggrader import ROOT, g, run_batch, write_csv


def u8(f):
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


def to3(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_tasks():
    src = g.imread(os.path.join(ROOT, "case_studies", "isic", "expected", "lesion.jpg"))
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    coffee = g.imread(os.path.join(ROOT, "test_kit", "generator", "sources", "skimage_coffee.png"))
    t = {}

    # 1. Grey-level conversion (ITU-R BT.601)
    t["gray"] = dict(ref=to3(gray), students=[
        ("Pillow convert('L')", True, to3(np.array(Image.fromarray(rgb).convert("L")))),
        ("NumPy BT.601 formula", True, to3(u8(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]))),
        ("RGB weights applied to BGR data", False, to3(u8(0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]))),
        ("green channel only", False, to3(src[..., 1].copy()))])

    # 2. Gaussian blur, sigma = 2
    t["blur"] = dict(ref=cv2.GaussianBlur(src, (0, 0), 2), students=[
        ("SciPy gaussian_filter", True, u8(np.stack([ndi.gaussian_filter(src[..., c].astype(float), 2) for c in range(3)], -1))),
        ("scikit-image gaussian", True, u8(255 * filters.gaussian(src, sigma=2, channel_axis=-1))),
        ("Pillow GaussianBlur(2)", True, cv2.cvtColor(np.array(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(2))), cv2.COLOR_RGB2BGR)),
        ("sigma 6 instead of 2", False, cv2.GaussianBlur(src, (0, 0), 6)),
        ("5x5 median filter instead of Gaussian", False, cv2.medianBlur(src, 5))])

    # 3. Otsu threshold (dark lesion = foreground)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w0, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    sb = np.where((w0 > 0) & (w0 < 1), (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12), 0)
    t["otsu"] = dict(ref=to3(otsu), students=[
        ("scikit-image threshold_otsu", True, to3(((gray <= filters.threshold_otsu(gray)) * 255).astype(np.uint8))),
        ("Otsu written with NumPy", True, to3(((gray <= np.argmax(sb)) * 255).astype(np.uint8))),
        ("fixed threshold 127", False, to3(((gray <= 127) * 255).astype(np.uint8))),
        ("foreground not inverted", False, to3(255 - otsu))])

    # 4. Morphological opening of the Otsu mask, 5x5 ellipse
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = otsu > 0
    t["opening"] = dict(ref=to3(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k)), students=[
        ("scikit-image opening, disk(2)", True, to3((morphology.opening(m, morphology.disk(2)) * 255).astype(np.uint8))),
        ("SciPy binary_opening, 5x5 ellipse", True, to3((ndi.binary_opening(m, structure=k.astype(bool)) * 255).astype(np.uint8))),
        ("closing instead of opening", False, to3(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k))),
        ("erosion only", False, to3(cv2.erode(otsu, k)))])

    # 5. Increase saturation by 30 % in HSV
    hsv = cv2.cvtColor(coffee, cv2.COLOR_BGR2HSV).astype(float)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
    ref = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h2 = color.rgb2hsv(cv2.cvtColor(coffee, cv2.COLOR_BGR2RGB) / 255.0)
    h2[..., 1] = np.clip(h2[..., 1] * 1.3, 0, 1)
    over = cv2.cvtColor(coffee, cv2.COLOR_BGR2HSV)
    over[..., 1] = (over[..., 1].astype(np.uint16) * 13 // 10).astype(np.uint8)      # wraps around above 255
    val = cv2.cvtColor(coffee, cv2.COLOR_BGR2HSV).astype(float)
    val[..., 2] = np.clip(val[..., 2] * 1.3, 0, 255)
    t["saturation"] = dict(ref=ref, students=[
        ("scikit-image rgb2hsv / hsv2rgb", True, cv2.cvtColor(u8(255 * color.hsv2rgb(h2)), cv2.COLOR_RGB2BGR)),
        ("8-bit overflow (no clipping)", False, cv2.cvtColor(over, cv2.COLOR_HSV2BGR)),
        ("value channel scaled instead", False, cv2.cvtColor(val.astype(np.uint8), cv2.COLOR_HSV2BGR))])

    # 6. Sobel gradient magnitude, scaled to 0..255
    def norm(a):
        return u8(255 * a / a.max())
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gf = gray.astype(float)
    t["sobel"] = dict(ref=to3(norm(np.hypot(gx, gy))), students=[
        ("scikit-image sobel", True, to3(norm(filters.sobel(gf)))),
        ("SciPy ndimage.sobel", True, to3(norm(np.hypot(ndi.sobel(gf, 1), ndi.sobel(gf, 0))))),
        ("x-gradient only", False, to3(norm(np.abs(gx)))),
        ("Laplacian instead of Sobel", False, to3(norm(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))))])
    return t


# (measure, points, tolerances, mask or None) per task and rubric
RUBRICS = {
    "initial": {
        "gray": [("SSIM", 2, "0.95,0.9,0.8", None), ("MAE", 1, "1,5,10,50", None)],
        "blur": [("SSIM", 2, "0.95,0.9,0.8", None), ("PSNR", 1, "30,25,20,15", None)],
        "otsu": [("IoU", 3, "0.99,0.97,0.95", None), ("Objects", 1, "0,2,5", None)],
        "opening": [("IoU", 3, "0.99,0.97,0.95", None), ("Euler", 1, "0,5,20", None)],
        "saturation": [("ΔE", 2, "1,2,5,10", None), ("SSIM", 1, "0.95,0.9,0.8", None)],
        "sobel": [("NCC", 2, "0.98,0.95,0.9", None), ("SSIM", 1, "0.9,0.8,0.7", None)],
    },
}
RUBRICS["calibrated"] = dict(RUBRICS["initial"])
RUBRICS["calibrated"].update({
    "blur": [("SSIM", 2, "0.995,0.99,0.98", None), ("PSNR", 1, "44,40,35", None)],
    "opening": [("IoU", 3, "0.998,0.995,0.99", None), ("Euler", 1, "0,5,20", None)],
    "sobel": [("NCC", 2, "0.98,0.95,0.9", "no_border.png"), ("SSIM", 1, "0.99,0.95,0.9", "no_border.png")],
})


def main():
    tasks = build_tasks()
    rows = []
    tmp = tempfile.mkdtemp()
    try:
        exp, stud = os.path.join(tmp, "expected"), os.path.join(tmp, "students")
        os.makedirs(exp)
        os.makedirs(stud)
        h, w = tasks["sobel"]["ref"].shape[:2]
        border = np.full((h, w), 255, np.uint8)
        border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = 0
        g.imwrite(os.path.join(tmp, "no_border.png"), border)
        for name, d in tasks.items():
            g.imwrite(os.path.join(exp, f"{name}.png"), d["ref"])
            for i, (_, _, img) in enumerate(d["students"]):
                assert img.shape == d["ref"].shape, name
                os.makedirs(os.path.join(stud, f"{name}__{i}"))
                g.imwrite(os.path.join(stud, f"{name}__{i}", f"{name}_stud.png"), img)
        for rubric, spec in RUBRICS.items():
            rules = [{"output": f"{t}_stud.png", "metric": m, "max_score": pts, "tolerances": tol,
                      "mode": "bucket", "aggregation": "mean", "notes": f"mask={mask}" if mask else ""}
                     for t in tasks for (m, pts, tol, mask) in spec[t]]
            res = {r["student"]: r for r in run_batch(stud, exp, rules, tmp)}
            for t, d in tasks.items():
                idx = [j for j, r in enumerate(rules, 1) if r["output"] == f"{t}_stud.png"]
                for i, (sname, ok, _) in enumerate(d["students"]):
                    r = res[f"{t}__{i}"]
                    keys = [f"R{j}_{rules[j - 1]['metric']}" for j in idx]
                    rows.append({"rubric": rubric, "task": t, "submission": sname, "correct": int(ok),
                                 "points": round(sum(float(r[k + "_score"]) for k in keys), 6),
                                 "max_points": sum(rules[j - 1]["max_score"] for j in idx),
                                 "values": "; ".join(f"{rules[j - 1]['metric']}={float(r[k + '_val']):.6g}"
                                                     for j, k in zip(idx, keys))})
    finally:
        shutil.rmtree(tmp)
    for rubric in RUBRICS:
        sep = []
        for t in tasks:
            sub = [r for r in rows if r["rubric"] == rubric and r["task"] == t]
            worst_ok = min(r["points"] for r in sub if r["correct"])
            best_bad = max(r["points"] for r in sub if not r["correct"])
            sep.append(worst_ok > best_bad)
            print(f"{rubric:10s} {t:10s} every correct solution above every mistake: {'yes' if sep[-1] else 'NO'}"
                  f"  (worst correct {worst_ok:g}, best mistake {best_bad:g})")
        print(f"{rubric}: {sum(sep)}/{len(sep)} tasks separated")
    print("wrote", write_csv("realistic_tasks.csv", rows))


if __name__ == "__main__":
    main()
