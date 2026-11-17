"""Build an independent test course for ImgGrader and its answer key.

Every student is designed for one evaluation scenario. The answer key is
computed by oracle.py (documentation-based, no grader code). Where a value is
known by construction (e.g. +5 on every channel -> MSE = 25), it is stored as
'designed' and the oracle is checked against it too.
Usage: python build_course.py <sources_dir> <course_dir>
"""
import os, sys, json, math, shutil
import numpy as np, cv2
import oracle as O

SRC, C = sys.argv[1], sys.argv[2]
shutil.rmtree(C, ignore_errors=True); os.makedirs(C)
rng = np.random.default_rng(2026)
def src(name): return cv2.imread(os.path.join(SRC, name), cv2.IMREAD_COLOR)
def to3(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def clip(x): return np.clip(np.round(x), 0, 255).astype(np.uint8)
def compress(img, lo=30, hi=225):                 # keeps +-20 changes away from 0/255 (no clipping)
    return clip(lo + img.astype(np.float64) * (hi - lo) / 255.0)

TASKS = {}
def task(name, expected_files, rules, students, explicit=None):
    """students: {student: {"files": {relpath: image|bytes|None}, "designed": {rule_idx: value}, "status": {rule_idx: str}}}"""
    T = os.path.join(C, name); os.makedirs(os.path.join(T, "expected")); os.makedirs(os.path.join(T, "students"))
    for fn, img in expected_files.items():
        cv2.imwrite(os.path.join(T, "expected", fn), img)
    for s, spec in students.items():
        d = os.path.join(T, "students", s); os.makedirs(d, exist_ok=True)
        for rel, obj in spec.get("files", {}).items():
            p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
            if isinstance(obj, (bytes, str)):
                open(p, "wb").write(obj if isinstance(obj, bytes) else obj.encode("utf-8"))
            else:
                ok, buf = cv2.imencode(os.path.splitext(p)[1], obj, spec.get("params", {}).get(rel, []))
                assert ok; buf.tofile(p)
    json.dump(rules, open(os.path.join(T, "rules.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    TASKS[name] = dict(rules=rules, students=students, explicit=explicit, expected_files=list(expected_files))
    return T

def rule(output, metric, mx, tol, mode="bucket", agg="mean", notes=""):
    return {"output": output, "metric": metric, "max_score": mx, "tolerances": tol, "mode": mode, "aggregation": agg, "notes": notes}

# ===================================================================== T01 pixel errors (exact, analytic)
ref = compress(src("skimage_astronaut.png")); N = ref.shape[0] * ref.shape[1]; o = "astro_stud.png"
R = [rule(o, "MSE", 4, "1,4,25,100"), rule(o, "MAE", 4, "1,2,5,10"), rule(o, "RMSE", 4, "1,2,5,10"),
     rule(o, "PSNR", 4, "40,34,28,20"), rule(o, "MSE", 2, "100", "continuous"), rule(o, "PSNR", 2, "48", "continuous"),
     rule(o, "EXACT", 1, ""), rule(o, "N_DIFF_PIXELS", 3, "thr:10 | 0,50,500", notes="logic=any"),
     rule(o, "N_DIFF_PIXELS", 3, "thr:10 | 0,50,500", notes="logic=all"), rule(o, "N_DIFF_PIXELS", 3, "thr:5 | 0,50,500", notes="gray=1")]
S = {"perfect": {"files": {o: ref}, "designed": {1: 0, 2: 0, 3: 0, 4: float("inf"), 7: 1, 8: 0, 9: 0}}}
for dd in (1, 2, 5, 10, 20):
    S[f"offset_plus{dd}"] = {"files": {o: (ref.astype(int) + dd).astype(np.uint8)},
                             "designed": {1: dd * dd, 2: dd, 3: dd, 4: 20 * math.log10(255 / dd), 7: 0,
                                          8: N if dd >= 10 else 0, 9: N if dd >= 10 else 0}}
k_img = ref.copy(); ys, xs = rng.choice(ref.shape[0], 37, replace=False), rng.choice(ref.shape[1], 37, replace=False)
k_img[ys, xs] = (k_img[ys, xs].astype(int) + 25).astype(np.uint8)
S["37_pixels_plus25"] = {"files": {o: k_img}, "designed": {1: 37 * 625 / N, 8: 37, 9: 37}}
b_img = ref.copy(); b_img[ys, xs, 0] = (b_img[ys, xs, 0].astype(int) + 25).astype(np.uint8)
S["37_pixels_blue_only"] = {"files": {o: b_img}, "designed": {8: 37, 9: 0}}
one = ref.copy(); one[10, 10, 2] += 1
S["one_value_off_by_1"] = {"files": {o: one}, "designed": {7: 0, 8: 0, 1: 1 / (3 * N)}}
task("T01_pixel_errors", {"astro.png": ref}, R, S)

# ===================================================================== T02 perceptual and correlation
ref = src("skimage_coffee.png"); o = "coffee_stud.png"
R = [rule(o, "SSIM", 3, "0.95,0.9,0.8"), rule(o, "SSIM", 2, "0.95", "continuous"), rule(o, "NCC", 3, "0.99,0.95,0.9"),
     rule(o, "NCC", 2, "0.99", "continuous"), rule(o, "PSNR", 2, "40,30,25")]
jpg = lambda q: cv2.imdecode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
S = {"perfect": {"files": {o: ref}, "designed": {1: 1.0, 3: 1.0, 5: float("inf")}},
     "blur_sigma1": {"files": {o: cv2.GaussianBlur(ref, (0, 0), 1)}}, "blur_sigma3": {"files": {o: cv2.GaussianBlur(ref, (0, 0), 3)}},
     "jpeg_q90": {"files": {o: jpg(90)}}, "jpeg_q20": {"files": {o: jpg(20)}},
     "noise_sigma5": {"files": {o: clip(ref + rng.normal(0, 5, ref.shape))}},
     "contrast_half": {"files": {o: clip(128 + 0.5 * (ref.astype(float) - 128))}},
     "inverted": {"files": {o: 255 - ref}, "designed": {3: -1.0}}}
task("T02_perceptual", {"coffee.png": ref}, R, S)

# ===================================================================== T03 colour difference (CIEDE2000)
ref = src("skimage_chelsea.png"); o = "chelsea_stud.png"
R = [rule(o, "ΔE", 4, "1,2,5,10"), rule(o, "ΔE", 2, "10", "continuous"), rule(o, "ΔE", 2, "2,8", agg="max")]
hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV).astype(int); hsv[..., 0] = (hsv[..., 0] + 5) % 180
red = ref.astype(int); red[..., 2] += 12
S = {"perfect": {"files": {o: ref}, "designed": {1: 0, 2: 0, 3: 0}},
     "red_cast_plus12": {"files": {o: clip(red)}}, "hue_shift_10deg": {"files": {o: cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)}},
     "grayscale": {"files": {o: to3(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))}}, "jpeg_q30": {"files": {o: cv2.imdecode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1)}}}
task("T03_colour", {"chelsea.png": ref}, R, S)

# ===================================================================== T04 aggregation (exact: 10000 px, rows changed)
ref = np.zeros((100, 100, 3), np.uint8); ref[:] = (120, 140, 160); o = "patch_stud.png"
lab = lambda bgr: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.array(bgr[::-1], float) / 255.0))
import colour
e = float(colour.delta_E(lab((120, 140, 160)), lab((120, 140, 190)), method="CIE 2000"))
R = []
for m, unit in (("MAE", 10.0), ("MSE", 300.0), ("ΔE", e)):
    for agg in ("mean", "median", "max", "trim10"):
        R.append(rule(o, m, 1, f"{0.1*unit:.6g},{0.5*unit:.6g},{0.9*unit:.6g}", agg=agg))
S = {}
for rows in (5, 30, 60):
    img = ref.copy(); img[:rows] = (120, 140, 190)
    p = rows * 100; zeros = 10000 - p
    def trim(z, q, k=1000):
        """Sorted values: z zeros then q ones; drop k lowest and k highest."""
        ones_left = q - max(0, k - z) - min(q, k)
        return max(ones_left, 0) / (z + q - 2 * k)
    frac = {"mean": p / 10000, "median": 1.0 if p > 5000 else 0.0, "max": 1.0, "trim10": trim(zeros, p)}
    S[f"rows_{rows}_changed"] = {"files": {o: img},
                                 "designed": {i + 1: frac[r["aggregation"]] * {"MAE": 10.0, "MSE": 300.0, "ΔE": e}[r["metric"]] for i, r in enumerate(R)}}
task("T04_aggregation", {"patch.png": ref}, R, S)

# ===================================================================== T05 masks and topology
h = cv2.imread(os.path.join(SRC, "skimage_horse.png"), 0) > 127
horse = (~h if h.mean() > 0.5 else h).astype(np.uint8) * 255      # horse = white
ref = to3(horse); o = "horse_stud.png"
from scipy import ndimage
dt = ndimage.distance_transform_edt(horse > 0)
assert dt.max() > 14, dt.max()
cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
k3 = np.ones((3, 3), np.uint8)
ring = cv2.dilate(horse, k3, iterations=3) - cv2.erode(horse, k3, iterations=3)
os.makedirs(os.path.join(C, "T05_masks_topology", "masks"), exist_ok=True)
R = [rule(o, "IoU", 3, "0.99,0.95,0.9"), rule(o, "IoU", 2, "1", "continuous"), rule(o, "Objects", 2, "0,1,3"),
     rule(o, "Holes", 2, "0,1"), rule(o, "Euler", 2, "0,1,2"), rule(o, "Contours", 1, "1"), rule(o, "ContoursStrict", 2, "1,0.5"),
     rule(o, "N_DIFF_PIXELS", 2, "thr:10 | 0,500,2000", notes="gray=1; mask=./masks/horse_ring.png")]
def blobs(img):
    assert img[2:18, 2:18].max() == 0 and img[-18:-2, 2:18].max() == 0      # corners are background
    out = img.copy(); out[5:15, 5:15] = 255; out[-15:-5, 5:15] = 255; return out
def hole(img, r=6):
    out = img.copy(); cv2.circle(out, (int(cx), int(cy)), r, 0, -1); return out
def island(img):
    out = hole(img, 10); cv2.circle(out, (int(cx), int(cy)), 4, 255, -1); return out
def split(img):
    out = img.copy(); out[:, int(cx) - 1:int(cx) + 2] = 0; return out
S = {"perfect": {"files": {o: ref}, "designed": {1: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0}},
     "eroded_3px": {"files": {o: to3(cv2.erode(horse, k3, iterations=3))}},
     "dilated_3px": {"files": {o: to3(cv2.dilate(horse, k3, iterations=3))}},
     "shifted_5px": {"files": {o: to3(np.roll(horse, 5, axis=1))}},
     "two_extra_blobs": {"files": {o: to3(blobs(horse))}, "designed": {3: 2, 4: 0, 5: 2}},
     "one_extra_hole": {"files": {o: to3(hole(horse))}, "designed": {3: 0, 4: 1, 5: 1, 6: 0}},
     "island_in_hole": {"files": {o: to3(island(horse))}, "designed": {3: 1, 4: 1, 5: 0, 6: 0}},
     "split_in_two": {"files": {o: to3(split(horse))}},
     "inverted": {"files": {o: to3(255 - horse)}}, "empty": {"files": {o: np.zeros_like(ref)}, "designed": {1: 0.0}},
     "mask_saved_as_0_and_1": {"files": {o: to3((horse > 0).astype(np.uint8))}, "designed": {1: 0.0}},
     "soft_antialiased_edges": {"files": {o: to3(cv2.GaussianBlur(horse, (0, 0), 2))}}}
T = task("T05_masks_topology", {"horse.png": ref}, R, S)
cv2.imwrite(os.path.join(T, "masks", "horse_ring.png"), ring)

# ===================================================================== T06 regions of interest
ref = compress(src("skimage_rocket.png")); H, W = ref.shape[:2]; o = "rocket_stud.png"
left = np.zeros((H, W), np.uint8); left[:, :W // 2] = 255; right = 255 - left
circ = np.zeros((H, W), np.uint8); cv2.circle(circ, (W // 2, H // 2), 120, 255, -1)
R = [rule(o, "MSE", 2, "1,100", notes="mask=./masks/left.png"), rule(o, "MSE", 2, "1,100", notes="mask=./masks/right.png"),
     rule(o, "N_DIFF_PIXELS", 2, "thr:10 | 0,1000", notes="mask=./masks/left.png"),
     rule(o, "N_DIFF_PIXELS", 2, "thr:10 | 0,1000", notes="mask=./masks/right.png"),
     rule(o, "MSE", 2, "1,100", notes="mask=./masks/left_half_resolution.png"),
     rule(o, "SSIM", 2, "0.99,0.9", notes="mask=./masks/circle.png"), rule(o, "MSE", 2, "1,100", notes="mask=./masks/empty.png"),
     rule(o, "MSE", 2, "1,100", notes="mask=./masks/does_not_exist.png"), rule(o, "MSE", 2, "1,250"),
     rule(o, "EXACT", 1, "", notes="mask=./masks/left.png"), rule(o, "PSNR", 2, "40", notes="mask=./masks/right.png"),
     rule(o, "N_DIFF_PIXELS", 2, "thr:10 | 0,20000", notes="mask=./masks/circle_half_resolution.png"),
     rule(o, "MSE", 2, "1,100", notes="mask=./masks/corrupt_mask.png")]
re_ = ref.astype(int); re_[:, W // 2:] += 20; le_ = ref.astype(int); le_[:, :W // 2] += 20
NR = H * (W - W // 2); NL = H * (W // 2)
everywhere = (ref.astype(int) + 20).astype(np.uint8)
NR = H * (W - W // 2); NL = H * (W // 2)
S = {"perfect": {"files": {o: ref}, "designed": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 9: 0, 10: 1}},
     "error_everywhere": {"files": {o: everywhere}, "designed": {1: 400, 2: 400, 3: NL, 4: NR, 9: 400}},
     "error_right_half": {"files": {o: re_.astype(np.uint8)}, "designed": {1: 0, 2: 400, 3: 0, 4: NR, 5: 0, 9: 400 * NR / (H * W), 10: 1, 11: 20 * math.log10(255 / 20)}},
     "error_left_half": {"files": {o: le_.astype(np.uint8)}, "designed": {1: 400, 2: 0, 3: NL, 4: 0, 5: 400, 10: 0, 11: float("inf")}}}
T = task("T06_roi", {"rocket.png": ref}, R, S)
os.makedirs(os.path.join(T, "masks"))
for n, m in [("left", left), ("right", right), ("circle", circ), ("empty", np.zeros_like(left)),
             ("left_half_resolution", cv2.resize(left, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)),
             ("circle_half_resolution", cv2.resize(circ, (W // 2, H // 2), interpolation=cv2.INTER_AREA))]:
    cv2.imwrite(os.path.join(T, "masks", n + ".png"), m)
open(os.path.join(T, "masks", "corrupt_mask.png"), "wb").write(b"not a PNG mask")

# ===================================================================== T07 files and formats
ref = src("skimage_astronaut_480x512.png"); o = "photo_stud.png"
R = [rule(o, "MSE", 2, "1,50,500"), rule(o, "SIZE", 1, "0,5,50"), rule(o, "EXACT", 1, "")]
p16 = (cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257)
S = {"exact_name": {"files": {o: ref}, "designed": {1: 0, 2: 0, 3: 1}},
     "jpeg_instead_of_png": {"files": {"photo_stud.jpg": ref}, "params": {"photo_stud.jpg": [cv2.IMWRITE_JPEG_QUALITY, 95]}},
     "suffix_student": {"files": {"photo_student.png": ref}, "designed": {1: 0, 3: 1}},
     "suffix_dash_stud": {"files": {"photo-stud.png": ref}, "designed": {1: 0, 3: 1}},
     "ambiguous_two_files": {"files": {"photo_student.png": ref, "photo-stud.png": ref}, "status": "AMBIGUOUS student"},
     "substring_trap_photo10": {"files": {"photo10_stud.png": ref}, "status": "MISSING student"},
     "missing_file": {"files": {"notes.txt": "no image here"}, "status": "MISSING student"},
     "corrupt_file": {"files": {o: b"this is not a PNG file"}, "status": "UNREADABLE student"},
     "zero_byte_file": {"files": {o: b""}, "status": "UNREADABLE student"},
     "grayscale_png": {"files": {o: cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)}},
     "rgba_png": {"files": {o: cv2.cvtColor(ref, cv2.COLOR_BGR2BGRA)}, "designed": {1: 0, 3: 1}},
     "png_16bit_gray": {"files": {o: p16}},
     "half_size": {"files": {o: cv2.resize(ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)}, "designed": {2: 256}},
     "larger_size": {"files": {o: cv2.resize(ref, (600, 560), interpolation=cv2.INTER_CUBIC)}, "designed": {2: 88}},
     "file_in_subfolder": {"files": {"output/" + o: ref}, "designed": {1: 0, 3: 1}},
     "Kovács Ödön": {"files": {o: ref}, "designed": {1: 0, 3: 1}},
     "student with spaces": {"files": {o: ref}, "designed": {1: 0, 3: 1}}}
task("T07_files_formats", {"photo_gt.png": ref}, R, S)          # reference named *_gt.png

# ===================================================================== T08 console output
EXP_MULTI = "Area: 1520\nPerimeter: 160.5"
R = [rule("cout", "exact", 2, EXP_MULTI, notes="file=stdout.txt"), rule("cout", "contains", 1, "Perimeter: 160.5"),
     rule("cout", "regex", 1, r"Area:\s*15\d\d\b"), rule("cout", "regex", 1, r"Area: ("),
     rule("cout", "contains", 1, "Area: 1520, Perimeter", notes=""), rule("cout", "exact", 1, "done", notes="file=log.txt"),
     rule("cout", "contains", 1, "Eredmény: 42")]
good = EXP_MULTI + "\n"
S = {"exact_output": {"files": {"stdout.txt": good, "log.txt": "done\n"}},
     "extra_spaces_and_newlines": {"files": {"stdout.txt": "\n  " + EXP_MULTI + "  \n\n"}},
     "windows_line_endings": {"files": {"stdout.txt": good.replace("\n", "\r\n").encode()}},
     "extra_debug_line": {"files": {"stdout.txt": "debug: start\n" + good}},
     "wrong_area": {"files": {"stdout.txt": "Area: 1521\nPerimeter: 160.5\n"}},
     "one_line_with_comma": {"files": {"stdout.txt": "Area: 1520, Perimeter: 160.5\n"}},
     "hungarian_text": {"files": {"stdout.txt": good + "Eredmény: 42 ✓\n"}},
     "empty_output": {"files": {"stdout.txt": ""}},
     "no_output_file": {"files": {"result.png": np.zeros((5, 5, 3), np.uint8)}}}
task("T08_console", {}, R, S)

# ===================================================================== T09 rule-file robustness
ref = src("skimage_clock.png"); o = "clock_stud.png"
R = [rule(o, "MSE", 1.5, "1,100"), rule(o, "MSE", 1.5, "1,400", agg="median"),                 # same metric twice, decimals
     rule(o, "SSIMM", 1, "0.9"), rule(o, "MSE", 1, ""), rule(o, "IoU", 1, "0.9;0.8"),            # unknown metric / empty / invalid tolerance
     rule("", "SIZE", 1, "0"),                                                                    # blank output: reuse previous file
     rule(o, "PSNR", 2, "—", "bucket"), rule(o, "EXACT", 1, "—")]                        # dash written as in the App-Guide
noisy = clip(ref + rng.normal(0, 8, ref.shape))
S = {"perfect": {"files": {o: ref}, "designed": {1: 0, 2: 0, 6: 0}, "status_by_rule": {3: "ERROR", 4: "ERROR", 5: "ERROR", 7: "ERROR"}},
     "noisy": {"files": {o: noisy}, "status_by_rule": {3: "ERROR", 4: "ERROR", 5: "ERROR", 7: "ERROR"}}}
task("T09_rule_files", {"clock.png": ref}, R, S)

json.dump({k: {**v, "students": {s: {kk: vv for kk, vv in sp.items() if kk not in ("files", "params")} | {"filenames": list(sp.get("files", {}))}
                                   for s, sp in v["students"].items()}} for k, v in TASKS.items()},
          open(os.path.join(C, "design.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print({k: (len(v["rules"]), len(v["students"])) for k, v in TASKS.items()})
