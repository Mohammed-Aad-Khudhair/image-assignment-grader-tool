"""Independent oracle for ImgGrader v1.1.

Written from the documentation only (User Guide, App-Guide, paper appendix and
README); it does NOT import or copy the grader. Library conventions that the
documentation names are used directly: OpenCV for reading images, grayscale
conversion and resizing; colour-science for CIEDE2000; SciPy for labelling.
"""
import math, re
import numpy as np, cv2
from scipy import ndimage as ndi
import colour

HIGHER_IS_BETTER = {"PSNR", "SSIM", "NCC", "IoU", "Contours", "ContoursStrict", "EXACT"}   # App-Guide §5
MAP_METRICS = {"MSE", "MAE", "ΔE"}                                                     # App-Guide §4


class Expected(Exception):
    """An evaluation the documentation says must be reported, not scored."""
    def __init__(self, status):
        super().__init__(status); self.status = status


# ---------------------------------------------------------------- helpers
def read(path, flags=cv2.IMREAD_COLOR):
    import os
    data = np.fromfile(path, dtype=np.uint8) if path and os.path.isfile(path) else None
    return None if data is None or data.size == 0 else cv2.imdecode(data, flags)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binary(img):
    return gray(img) > 127                     # masks/IoU: white = foreground (>127)

def roi_from_file(path, shape_hw):
    m = read(path, cv2.IMREAD_GRAYSCALE) if path else None
    if m is None:
        raise Expected("ERROR")                # missing mask is an error (paper §2)
    if m.shape != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    roi = m > 127
    if not roi.any():
        raise Expected("ERROR")                # empty mask
    return roi

def align(exp, stud, metric):
    """Student resized to the reference size with area interpolation, except SIZE."""
    if metric != "SIZE" and stud.shape[:2] != exp.shape[:2]:
        return cv2.resize(stud, (exp.shape[1], exp.shape[0]), interpolation=cv2.INTER_AREA), 1
    return stud, 0

def aggregate(values, how):
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if how in ("", "mean", None): return float(v.mean())
    if how == "median": return float(np.median(v))
    if how == "max": return float(v.max())
    m = re.fullmatch(r"trim(\d+)", how)
    if m:                                       # drop p% lowest and p% highest, then mean
        k = int(len(v) * int(m.group(1)) / 100)
        return float(v[k:len(v) - k].mean()) if len(v) - 2 * k > 0 else float(v.mean())
    raise ValueError(how)

def ndiff_params(tol, notes):
    """N_DIFF tolerances 'thr:<t> | c1,c2'; notes logic=any|all, gray=1."""
    m = re.search(r"thr\s*[:=]\s*(\d+)", tol); thr = int(m.group(1)) if m else 10
    bands = tol.split("|", 1)[1] if "|" in tol else re.sub(r"thr\s*[:=]\s*\d+", "", tol)
    logic = "all" if re.search(r"logic\s*=\s*all", notes or "") else "any"
    use_gray = bool(re.search(r"gr[ae]y\s*=\s*(1|true)", notes or "", re.I))
    return thr, logic, use_gray, bands.strip()

# ---------------------------------------------------------------- SSIM (Wang et al. 2004)
def ssim_map(a, b, win=7):
    """Uniform 7x7 window, sample covariance, K1=0.01, K2=0.03, L=255 (scikit-image defaults)."""
    x, y = gray(a).astype(np.float64), gray(b).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    n = win * win; cov = n / (n - 1.0)
    f = lambda z: ndi.uniform_filter(z, size=win, mode="reflect")
    ux, uy = f(x), f(y)
    vx, vy = cov * (f(x * x) - ux * ux), cov * (f(y * y) - uy * uy)
    vxy = cov * (f(x * y) - ux * uy)
    return ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))

# ---------------------------------------------------------------- topology
S8, S4 = np.ones((3, 3), bool), ndi.generate_binary_structure(2, 1)
def n_objects(img): return ndi.label(binary(img), structure=S8)[1]
def n_holes(img):
    bg = ~binary(img); lab, n = ndi.label(bg, structure=S4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return n - len(edge)
def holes_per_object(img):
    lab, _ = ndi.label(binary(img), structure=S8); out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        sub = np.pad(lab[sl] == i, 1)
        out.append(ndi.label(ndi.binary_fill_holes(sub, structure=S4) & ~sub, structure=S4)[1])
    return sorted(out)

# ---------------------------------------------------------------- metrics
def metric(name, exp, stud, roi=None, agg="mean", tol="", notes=""):
    """Value of a documented metric. exp/stud are aligned BGR uint8 images."""
    sel = (lambda m: m[roi]) if roi is not None else (lambda m: m.ravel())
    if name == "SIZE":
        return float(max(abs(exp.shape[1] - stud.shape[1]), abs(exp.shape[0] - stud.shape[0])))
    d = exp.astype(np.float64) - stud.astype(np.float64)
    if name == "MSE":  return aggregate(sel((d ** 2).mean(axis=2)), agg)
    if name == "MAE":  return aggregate(sel(np.abs(d).mean(axis=2)), agg)
    if name == "RMSE": return math.sqrt(sel((d ** 2).mean(axis=2)).mean())
    if name == "PSNR":
        mse = sel((d ** 2).mean(axis=2)).mean()
        return float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    if name == "ΔE":
        lab = lambda im: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0))
        return aggregate(sel(colour.delta_E(lab(exp), lab(stud), method="CIE 2000")), agg)
    if name == "SSIM":
        S = ssim_map(exp, stud)
        return float(S[roi].mean()) if roi is not None else float(S[3:-3, 3:-3].mean())
    if name == "NCC":
        x, y = sel(gray(exp).astype(np.float64)), sel(gray(stud).astype(np.float64))
        x, y = x - x.mean(), y - y.mean(); den = np.linalg.norm(x) * np.linalg.norm(y)
        return 0.0 if den == 0 else float(x @ y / den)
    if name == "IoU":
        A, B = sel(binary(exp)), sel(binary(stud)); u = (A | B).sum()
        return 1.0 if u == 0 else float((A & B).sum() / u)
    if name == "EXACT":
        if roi is not None:
            return 1.0 if np.array_equal(exp[roi], stud[roi]) else 0.0
        return 1.0 if np.array_equal(exp, stud) else 0.0
    if name == "N_DIFF_PIXELS":
        thr, logic, use_gray, _ = ndiff_params(tol, notes)
        if use_gray:
            bad = np.abs(gray(exp).astype(int) - gray(stud).astype(int)) >= thr
        else:
            ad = np.abs(d) >= thr
            bad = ad.all(axis=2) if logic == "all" else ad.any(axis=2)
        return float(sel(bad).sum())
    # shape metrics: pixels outside the ROI are set to background
    if roi is not None:
        exp = exp * roi[:, :, None].astype(np.uint8); stud = stud * roi[:, :, None].astype(np.uint8)
    if name == "Objects": return float(abs(n_objects(exp) - n_objects(stud)))
    if name == "Holes":   return float(abs(n_holes(exp) - n_holes(stud)))
    if name == "Euler":
        return float(abs((n_objects(exp) - n_holes(exp)) - (n_objects(stud) - n_holes(stud))))
    if name == "Contours":
        return 1.0 if n_objects(exp) + n_holes(exp) == n_objects(stud) + n_holes(stud) else 0.0
    if name == "ContoursStrict":
        from collections import Counter
        A, B = Counter(holes_per_object(exp)), Counter(holes_per_object(stud)); k = set(A) | set(B)
        return 1.0 if not k else sum(min(A[i], B[i]) for i in k) / sum(max(A[i], B[i]) for i in k)
    raise Expected("ERROR")                    # unknown metric

# ---------------------------------------------------------------- scoring (User Guide Table 2, App-Guide §5-6, paper §2)
def score(name, value, max_score, tol, mode):
    parts = [t.strip() for t in tol.split(",") if t.strip() not in ("", "-", "–", "—")]
    try:
        t = [float(p) for p in parts]
    except ValueError:
        raise Expected("ERROR")
    if not t:
        if name == "EXACT":
            return max_score if value >= 1 else 0.0
        raise Expected("ERROR")
    n = len(t)
    if name in HIGHER_IS_BETTER:
        t = sorted(t, reverse=True)
        if mode == "bucket":
            for i, ti in enumerate(t):
                if value >= ti: return max_score * (n - i) / n
            return 0.0
        return 0.0 if t[0] == 0 else max_score * min(1.0, max(0.0, value / t[0]))
    t = sorted(t)
    if mode == "bucket":
        for i, ti in enumerate(t):
            if value <= ti: return max_score * (n - i) / n
        return 0.0
    return 0.0 if t[-1] == 0 else max_score * max(0.0, 1.0 - value / t[-1])

def near_boundary(name, value, tol, eps):
    """True if the value lies within eps of a band threshold (score then depends on rounding)."""
    if name == "N_DIFF_PIXELS": tol = ndiff_params(tol, "")[3]
    try: t = [float(p) for p in tol.split(",") if p.strip()]
    except ValueError: return False
    return any(abs(value - ti) < eps and value != ti for ti in t) if math.isfinite(value) else False

# ---------------------------------------------------------------- console (App-Guide §2, §7)
def console(student_dir, metric_name, expected_text, notes):
    import os
    m = re.search(r"file\s*=\s*([^,;\s]+)", notes or ""); fname = m.group(1) if m else "stdout.txt"
    p = os.path.join(student_dir, fname)
    if not os.path.isfile(p):
        raise Expected(f"MISSING {fname}")
    with open(p, encoding="utf-8") as f:
        text = f.read().strip()
    e = expected_text.strip(); mode = metric_name.lower()
    if mode == "exact": return 1.0 if text == e else 0.0
    if mode == "contains": return 1.0 if e in text else 0.0
    if mode == "regex":
        try: return 1.0 if re.search(e, text) else 0.0
        except re.error: raise Expected("ERROR: invalid regex")
    raise Expected("ERROR")
