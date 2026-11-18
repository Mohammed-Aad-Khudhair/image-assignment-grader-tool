"""What does ImgGrader's automatic resizing do to values and scores?

A student image whose size differs from the reference is resized to the
reference size with area interpolation (cv2.INTER_AREA) before comparison.

1. Case studies (binary masks). Each student mask is saved at 75 %, 50 % and 25 %
   of the reference size, as if the student had worked at a lower resolution
   (nearest-neighbour downscaling keeps the mask binary). These smaller files
   are graded with ImgGrader exactly like the originals, and every rule value
   and score is compared with the full-size result.
   Output: results/resizing_case_studies.csv

2. Interpolation methods. For the same smaller masks and for grey-level images
   (test_kit/T02_perceptual), the image is scaled back with nearest, bilinear,
   bicubic and area interpolation and compared with the reference (IoU or MSE,
   SSIM). The change in the grey-level distribution caused by the round trip is
   measured with the chi-square distance between 256-bin histograms of the
   original-size image and the round-trip image:
   chi2 = 0.5 * sum((p - q)^2 / (p + q)), with p, q normalised histograms
   (0 = identical distributions, 1 = disjoint).
   Output: results/resizing_interpolation.csv
"""
import os
import shutil
import tempfile

import cv2
import numpy as np

from _imggrader import CASES, ROOT, g, load_rules, rule_keys, run_batch, write_csv

FACTORS = (0.75, 0.5, 0.25)
INTERPOLATIONS = [("nearest", cv2.INTER_NEAREST), ("bilinear", cv2.INTER_LINEAR),
                  ("bicubic", cv2.INTER_CUBIC), ("area (ImgGrader)", cv2.INTER_AREA)]


def shrink(img, f):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_NEAREST)


def chi2(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p = np.bincount(ga.ravel(), minlength=256) / ga.size
    q = np.bincount(gb.ravel(), minlength=256) / gb.size
    s = p + q
    nz = s > 0
    return float(0.5 * np.sum((p[nz] - q[nz]) ** 2 / s[nz]))


def case_studies():
    rows = []
    tmp = tempfile.mkdtemp()
    try:
        for name, case_dir, rule_file in CASES:
            rules = load_rules(os.path.join(case_dir, rule_file))
            keys = rule_keys(rules)
            out_name = rules[0]["output"]
            students = os.path.join(case_dir, "students")
            full = {r["student"]: r for r in run_batch(students, os.path.join(case_dir, "expected"), rules, case_dir)}
            for f in FACTORS:
                small_dir = os.path.join(tmp, f"{name}_{f}")
                for s in sorted(os.listdir(students)):
                    img = g.imread(os.path.join(students, s, out_name))
                    os.makedirs(os.path.join(small_dir, s))
                    g.imwrite(os.path.join(small_dir, s, out_name), shrink(img, f))
                small = run_batch(small_dir, os.path.join(case_dir, "expected"), rules, case_dir)
                for r in small:
                    s = r["student"]
                    for i, (k, rule) in enumerate(zip(keys, rules), 1):
                        assert r[f"{k}_resized"] == 1
                        v0, v1 = float(full[s][f"{k}_val"]), float(r[f"{k}_val"])
                        rows.append({"case": name, "student": s, "factor": f, "rule": i, "metric": rule["metric"],
                                     "max_score": rule["max_score"],
                                     "value_full": v0, "value_resized": v1, "value_change": v1 - v0,
                                     "score_full": float(full[s][f"{k}_score"]), "score_resized": float(r[f"{k}_score"]),
                                     "score_change": float(r[f"{k}_score"]) - float(full[s][f"{k}_score"])})
                for s in full:
                    t0 = float(full[s]["total_score"])
                    t1 = float(next(r for r in small if r["student"] == s)["total_score"])
                    rows.append({"case": name, "student": s, "factor": f, "rule": "total", "metric": "",
                                 "max_score": sum(r["max_score"] for r in rules),
                                 "value_full": "", "value_resized": "", "value_change": "",
                                 "score_full": t0, "score_resized": t1, "score_change": t1 - t0})
            print(name, "done")
    finally:
        shutil.rmtree(tmp)
    return rows


def interpolation():
    rows = []
    isic = os.path.join(ROOT, "case_studies", "isic")
    ref = g.imread(os.path.join(isic, "expected", "lesion_mask-reference.png"))
    items = [("ISIC", s, ref, g.imread(os.path.join(isic, "students", s, "lesion_mask_stud.png")), "IoU")
             for s in ("S2", "S3", "S4")]
    t02 = os.path.join(ROOT, "test_kit", "T02_perceptual")
    ref2 = g.imread(os.path.join(t02, "expected", "coffee.png"))
    items += [("T02 coffee", s, ref2, g.imread(os.path.join(t02, "students", s, "coffee_stud.png")), "MSE")
              for s in ("perfect", "noise_sigma5", "jpeg_q20", "blur_sigma1")]
    for case, s, exp, stud, m1 in items:
        h, w = stud.shape[:2]
        v_full = (g.compute_metric(m1, exp, stud)[0], g.compute_metric("SSIM", exp, stud)[0])
        for f in (0.5, 0.25):
            small = shrink(stud, f)
            for label, flag in INTERPOLATIONS:
                back = cv2.resize(small, (w, h), interpolation=flag)
                v = (g.compute_metric(m1, exp, back)[0], g.compute_metric("SSIM", exp, back)[0])
                rows.append({"case": case, "student": s, "factor": f, "interpolation": label, "metric": m1,
                             "value_full": v_full[0], "value_resized": v[0], "value_change": v[0] - v_full[0],
                             "ssim_full": v_full[1], "ssim_resized": v[1], "ssim_change": v[1] - v_full[1],
                             "chi2_histogram": chi2(stud, back)})
    return rows


if __name__ == "__main__":
    print("wrote", write_csv("resizing_case_studies.csv", case_studies()))
    print("wrote", write_csv("resizing_interpolation.csv", interpolation()))
