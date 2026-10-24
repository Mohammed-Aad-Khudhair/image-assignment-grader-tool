import sys, os, csv, json, math, re, tempfile
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QFormLayout, QLineEdit, QComboBox, QMessageBox,
    QSplitter, QProgressDialog, QSpinBox, QDoubleSpinBox, QTextEdit
)

from skimage.metrics import structural_similarity as sk_ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from collections import Counter

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------- Metrics ----------
def _ensure_same_size(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b

def metric_mse(a, b):
    a, b = _ensure_same_size(a, b)
    return float(np.mean((a.astype(float)-b.astype(float))**2))

def metric_mae(a, b):
    a, b = _ensure_same_size(a, b)
    return float(np.mean(np.abs(a.astype(float)-b.astype(float))))

def metric_rmse(a, b):
    return float(math.sqrt(metric_mse(a,b)))

def metric_psnr(a, b):
    m = metric_mse(a, b)
    if m == 0:
        return float('inf')
    return 20.0 * math.log10(255.0 / math.sqrt(m))

def metric_ssim(a, b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    try:
        return float(sk_ssim(a_gray, b_gray, data_range=255))
    except TypeError:
        return float(sk_ssim(a_gray, b_gray))

def metric_ncc(a, b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(float).ravel()
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(float).ravel()
    a_c = a_gray - a_gray.mean()
    b_c = b_gray - b_gray.mean()
    denom = np.linalg.norm(a_c)*np.linalg.norm(b_c)
    if denom == 0: return 0.0
    return float((a_c @ b_c) / denom)

def metric_iou(a, b, thresh=127):
    A = (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) > thresh).astype(np.uint8)
    B = (cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) > thresh).astype(np.uint8)
    inter = np.logical_and(A,B).sum()
    uni = np.logical_or(A,B).sum()
    return 1.0 if uni == 0 else float(inter/uni)

def metric_deltaE(a, b):
    a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(float)/255.0
    b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(float)/255.0
    a_lab = rgb2lab(a_rgb); b_lab = rgb2lab(b_rgb)
    dE = deltaE_ciede2000(a_lab, b_lab)
    return float(np.mean(dE))

def _count_objects(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    n, _ = cv2.connectedComponents(th)
    return max(0, int(n-1))

def _count_holes(img):
    # A hole is a background region enclosed by foreground (inner contour);
    # the outer background is not counted.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return int(sum(1 for h in hier[0] if h[3] != -1))

def metric_count_objects(a, b):
    # Objects miscount |C(A) - C(B)|, lower is better.
    return abs(_count_objects(a) - _count_objects(b))

def metric_count_holes(a, b):
    # Holes miscount |H(A) - H(B)|, lower is better.
    return abs(_count_holes(a) - _count_holes(b))

def metric_euler(a, b):
    # Euler-number difference |chi(A) - chi(B)| with chi = objects - holes,
    # lower is better. Catches topological errors (e.g. a spurious hole
    # inside a region) that leave IoU almost unchanged.
    chi_a = _count_objects(a) - _count_holes(a)
    chi_b = _count_objects(b) - _count_holes(b)
    return abs(chi_a - chi_b)

def metric_contour_hierarchy(a, b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, a_th = cv2.threshold(a_gray, 127, 255, cv2.THRESH_BINARY)
    _, b_th = cv2.threshold(b_gray, 127, 255, cv2.THRESH_BINARY)
    cnts_a, _ = cv2.findContours(a_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b, _ = cv2.findContours(b_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return 1.0 if len(cnts_a)==len(cnts_b) else 0.0

def metric_exact(a,b):
    a,b = _ensure_same_size(a,b)
    return 1.0 if np.array_equal(a,b) else 0.0

def metric_n_diff_pixels(a, b, thr=1, logic='any', use_gray=False):
    a, b = _ensure_same_size(a, b)
    if use_gray:
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ga - gb)
        return float((diff >= thr).sum())
    else:
        ad = np.abs(a.astype(np.float32) - b.astype(np.float32))
        if logic == 'all':
            mask = (ad[:,:,0] >= thr) & (ad[:,:,1] >= thr) & (ad[:,:,2] >= thr)
        else:
            mask = (ad[:,:,0] >= thr) | (ad[:,:,1] >= thr) | (ad[:,:,2] >= thr)
        return float(mask.sum())

def _holes_signature(gray):
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return []
    hier = hier[0]
    holes_per_outer = []
    for idx, h in enumerate(hier):
        if h[3] == -1:
            c = sum(1 for j, hh in enumerate(hier) if hh[3] == idx)
            holes_per_outer.append(c)
    holes_per_outer.sort()
    return holes_per_outer

def metric_contours_strict(a, b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sigA = _holes_signature(a_gray)
    sigB = _holes_signature(b_gray)
    if not sigA and not sigB:
        return 1.0
    cA = Counter(sigA); cB = Counter(sigB)
    inter = sum(min(cA[k], cB[k]) for k in set(cA)|set(cB))
    union = sum(max(cA[k], cB[k]) for k in set(cA)|set(cB))
    if union == 0: return 0.0
    return float(inter/union)

def metric_size_match(a, b, expect_w=None, expect_h=None):
    if expect_w is not None and expect_h is not None:
        return 1.0 if (b.shape[1] == expect_w and b.shape[0] == expect_h) else 0.0
    return 1.0 if (b.shape[1] == a.shape[1] and b.shape[0] == a.shape[0]) else 0.0

def parse_ndiff_params(tol_str, notes):
    thr = 10
    logic = 'any'
    use_gray = False
    tol_clean = (tol_str or "").strip()

    if tol_clean:
        m = re.search(r"thr\s*[:=]\s*(\d+)", tol_clean, re.I)
        if m:
            thr = int(m.group(1))
            tol_clean = re.sub(r"thr\s*[:=]\s*\d+\s*\|?\s*", "", tol_clean, flags=re.I).strip()
        if tol_clean:
            parts = [p.strip() for p in tol_clean.split(",") if p.strip()]
            tol_clean = ",".join(parts)

    if notes:
        m = re.search(r"thr\s*=\s*(\d+)", notes, re.I)
        if m: thr = int(m.group(1))
        m = re.search(r"logic\s*=\s*(any|all)", notes, re.I)
        if m: logic = m.group(1).lower()
        m = re.search(r"(gray|grey)\s*=\s*([01]|true|false)", notes, re.I)
        if m:
            use_gray = m.group(2).lower() in ('1','true')

    return thr, logic, use_gray, tol_clean

def read_console_text(student_dir, rule_notes):
    fname = "stdout.txt"
    if isinstance(rule_notes,str):
        m = re.search(r"file\s*=\s*([^,;\s]+)", rule_notes)
        if m: fname = m.group(1)
    p = os.path.join(student_dir, fname)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def console_compare(text, expected_text, mode="exact"):
    t = text.strip(); e = expected_text.strip()
    if mode == "exact": return 1.0 if t==e else 0.0
    if mode == "contains": return 1.0 if e in t else 0.0
    if mode == "regex":
        try: return 1.0 if re.search(e,t) else 0.0
        except Exception: return 0.0
    return 0.0

def parse_notes_kv(notes):
    """key=value pairs separated by ',' or ';'. A value may be double-quoted
    (key="a, b; c") to contain commas or semicolons."""
    out={}
    if not isinstance(notes,str): return out
    for m in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,;]+))', notes):
        val = m.group(2) if m.group(2) is not None else m.group(3)
        out[m.group(1).lower()] = val.strip() if m.group(2) is None else val
    return out

def expected_cout_text(tols, notes):
    """Expected console text: the Tolerance(s) field (as in the user guide);
    falls back to exp_cout=... in the notes for older rule files."""
    if tols and tols.strip():
        return tols
    return parse_notes_kv(notes).get("exp_cout", "")

def metric_size(a, b):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    return float(max(abs(wb - wa), abs(hb - ha)))

METRICS = {
    "MSE": metric_mse,
    "MAE": metric_mae,
    "RMSE": metric_rmse,
    "PSNR": metric_psnr,
    "SSIM": metric_ssim,
    "NCC": metric_ncc,
    "IoU": metric_iou,
    "ΔE": metric_deltaE,
    "Objects": metric_count_objects,
    "Holes": metric_count_holes,
    "Euler": metric_euler,
    "Contours": metric_contour_hierarchy,
    "ContoursStrict": metric_contours_strict,
    "EXACT": metric_exact,
    "N_DIFF_PIXELS": metric_n_diff_pixels,
    "SIZE": metric_size,
}

DEFAULT_TOLERANCES = {
    "MSE": "1,5,10,100",
    "MAE": "1,5,10,50",
    "RMSE": "1,2,5,10",
    "PSNR": "30,25,20,15",
    "SSIM": "0.95,0.9,0.8",
    "NCC": "0.95,0.9,0.8",
    "IoU": "0.95,0.9,0.8",
    "ΔE": "1,2,10,50",
    "Objects": "0",
    "Holes": "0",
    "Euler": "0",
    "Contours": "1",
    "ContoursStrict": "1,0.75,0.5",
    "EXACT": "1",
    "N_DIFF_PIXELS": "thr:10 | 100,1000,5000",
    "SIZE": "0,2,5,10"
}

FRIENDLY_METRICS = {
    "SIZE (dimensions)": "SIZE",
    "N_DIFF_PIXELS (pixel count)": "N_DIFF_PIXELS",
    "MSE": "MSE",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "PSNR": "PSNR",
    "SSIM": "SSIM",
    "NCC": "NCC",
    "IoU": "IoU",
    "ΔE": "ΔE",
    "Objects": "Objects",
    "Holes": "Holes",
    "Euler": "Euler",
    "Contours": "Contours",
    "ContoursStrict": "ContoursStrict",
    "EXACT": "EXACT"
}

def compute_score(metric_name, value, max_score, tol_str, mode="bucket"):
    try: max_score = float(max_score)
    except: max_score = 1.0
    try: tols = [float(t.strip()) for t in tol_str.split(",") if t.strip()]
    except: tols = []
    higher = metric_name in {"PSNR","SSIM","NCC","IoU","Contours","ContoursStrict","EXACT"}
    if not tols: return float(max_score)
    if higher:
        tols = sorted(tols, reverse=True)
        if mode=="bucket":
            for i,t in enumerate(tols):
                if value >= t: return max_score*(1.0 - i/len(tols))
            return 0.0
        top = tols[0]
        if top==0: return 0.0
        return max_score*min(value/top,1.0)
    else:
        tols = sorted(tols)
        if mode=="bucket":
            for i,t in enumerate(tols):
                if value <= t: return max_score*(1.0 - i/len(tols))
            return 0.0
        top = tols[-1]
        if top==0: return 0.0
        return max_score*max(0.0, 1.0 - value/top)

def map_expected_from_student(student_fname, expected_dir):
    name = os.path.basename(student_fname)
    base, ext = os.path.splitext(name)
    candidates = []
    for s in ("_stud","_student","-stud","-student"):
        if base.endswith(s):
            candidates.append(base[:-len(s)] + ext)
    candidates.append(base+ext)
    candidates.append(base.replace("_stud","").replace("_student","")+ext)
    for c in candidates:
        p = os.path.join(expected_dir, c)
        if os.path.exists(p): return p
    return ""

# ------ helpers for SIZE fallback parsing ------
def parse_expected_size_any(notes, tols):
    nd = parse_notes_kv(notes)
    ew = nd.get("expect_w") or nd.get("w") or nd.get("width")
    eh = nd.get("expect_h") or nd.get("h") or nd.get("height")
    if ew and eh:
        try: return int(float(ew)), int(float(eh))
        except: pass
    # also accept a WxH pair anywhere in notes or tolerances
    blob = f"{notes or ''} {tols or ''}"
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", blob)
    if m:
        try: return int(m.group(1)), int(m.group(2))
        except: pass
    return None, None

# ------------------ UI: Add Rule ------------------
class RulePopup(QDialog):
    def __init__(self, parent=None, autofill_output=""):
        super().__init__(parent)
        self.setWindowTitle("Add Rule")
        layout = QFormLayout(self)

        self.output_field = QLineEdit(autofill_output)
        self.metric_box = QComboBox(); self.metric_box.addItems(list(FRIENDLY_METRICS.keys()))
        self.agg_box = QComboBox(); self.agg_box.addItems(['mean','median','max','trim10'])
        self.output_field.textChanged.connect(self.on_output_change)
        self.max_score = QDoubleSpinBox(); self.max_score.setRange(0,100); self.max_score.setDecimals(2); self.max_score.setSingleStep(0.5); self.max_score.setValue(1)
        self.tol_edit = QLineEdit()
        self.cout_expected_edit = QLineEdit()
        self.notes_edit = QLineEdit()
        self.mode_box = QComboBox(); self.mode_box.addItems(["bucket","continuous"])

        self.metric_box.currentTextChanged.connect(self.on_metric_change)

        layout.addRow("Output (student filename):", self.output_field)
        layout.addRow("Metric:", self.metric_box)
        layout.addRow("Aggregation:", self.agg_box)
        layout.addRow("Max score:", self.max_score)
        self.row_tol = layout.addRow("Tolerance(s):", self.tol_edit)
        self.row_cout = layout.addRow("Expected console output:", self.cout_expected_edit)
        layout.addRow("Scoring mode:", self.mode_box)
        layout.addRow("Keywords/notes:", self.notes_edit)

        ok = QPushButton("Add"); ok.clicked.connect(self.accept)
        layout.addRow(ok)
        self.on_metric_change(self.metric_box.currentText())
        self.on_output_change(self.output_field.text())

    def on_output_change(self, txt):
        is_cout = (txt.strip().lower()=="cout")
        if is_cout:
            self.metric_box.clear(); self.metric_box.addItems(['exact','contains','regex'])
            self.tol_edit.clear()
        else:
            if self.metric_box.count()!=len(FRIENDLY_METRICS):
                self.metric_box.clear(); self.metric_box.addItems(list(FRIENDLY_METRICS.keys()))
        self.tol_edit.setVisible(not is_cout)
        self.cout_expected_edit.setVisible(is_cout)
        self.agg_box.setEnabled(False if is_cout else self.agg_box.isEnabled())

    def on_metric_change(self, metric_display):
        metric = FRIENDLY_METRICS.get(metric_display, metric_display)
        if metric in DEFAULT_TOLERANCES:
            self.tol_edit.setText(DEFAULT_TOLERANCES[metric])
        is_map = metric in ("ΔE","MAE","MSE")
        self.agg_box.setEnabled(is_map)
        self.agg_box.setVisible(is_map)

    def get_rule(self):
        notes_text = self.notes_edit.text().strip()
        out = self.output_field.text().strip()
        expected_cout = ""
        if out.lower()=="cout":
            # Stored in Tolerance(s) so commas/semicolons in the text are kept.
            expected_cout = self.cout_expected_edit.text().strip()
        return {
            "output": out,
            "metric": (self.metric_box.currentText().lower() if out.lower()=="cout"
                       else FRIENDLY_METRICS[self.metric_box.currentText()]),
            "max_score": float(self.max_score.value()),
            "tolerances": expected_cout if out.lower()=="cout" else self.tol_edit.text().strip(),
            "mode": self.mode_box.currentText(),
            "aggregation": self.agg_box.currentText() if self.agg_box.isEnabled() else "",
            "notes": notes_text,
            "expected_cout": expected_cout
        }

def _trimmed_mean(arr, trim_ratio=0.1):
    arr = np.asarray(arr).ravel(); n = len(arr)
    if n==0: return float('nan')
    k = int(n*trim_ratio)
    if k*2>=n: return float(np.mean(arr))
    s = np.sort(arr); return float(np.mean(s[k:n-k]))

def metric_deltaE_map(a,b):
    a_lab = rgb2lab(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    b_lab = rgb2lab(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    return deltaE_ciede2000(a_lab,b_lab)

def metric_mae_map(a,b):
    a,b = _ensure_same_size(a,b); return np.abs(a.astype(float)-b.astype(float)).mean(axis=2)

def metric_mse_map(a,b):
    a,b = _ensure_same_size(a,b); diff = (a.astype(float)-b.astype(float))**2; return diff.mean(axis=2)

def aggregate_map(m, mode="mean"):
    if mode=="mean": return float(np.mean(m))
    if mode=="median": return float(np.median(m))
    if mode=="max": return float(np.max(m))
    if mode.startswith("trim"):
        mm = re.search(r"(\d+)", mode); ratio = 0.1
        if mm: ratio = int(mm.group(1))/100.0
        return _trimmed_mean(m, ratio)
    return float(np.mean(m))

def _cout_fields(rule):
    """(notes, tolerances) for the table from a JSON rule. An expected_cout
    field (older files) is moved into Tolerance(s) and its exp_cout= prefix is
    removed from the notes, so text with commas is not truncated."""
    notes = rule.get("notes", "") or ""
    tol_text = rule.get("tolerances", "") or ""
    exp = rule.get("expected_cout") or ""
    if exp and (rule.get("output","").strip().lower() == "cout"):
        prefix = f"exp_cout={exp}"
        if notes.startswith(prefix):
            notes = notes[len(prefix):].lstrip(",; ")
        if not tol_text.strip():
            tol_text = exp
    return notes, tol_text

# ------------------ Shared metric evaluation (single + batch) ------------------
def compute_metric(metric, exp, stud, roi=None, aggregation="mean", tols="", notes=""):
    """Evaluate one metric. exp/stud are BGR images of equal size (except SIZE,
    which compares original dimensions). roi is a boolean HxW mask or None.

    With an ROI, every metric uses ROI pixels only: mean-based metrics
    (MSE, MAE, RMSE, PSNR, dE, SSIM, NCC) average over ROI pixels; counts
    (N_DIFF_PIXELS) and overlaps (IoU) are restricted to the ROI; shape metrics
    (Objects, Holes, Euler, Contours) use the images with non-ROI pixels set to 0.
    Returns (value, tolerances) - tolerances are cleaned for N_DIFF_PIXELS."""
    if metric == "SIZE":
        return float(metric_size(exp, stud)), tols
    if metric not in METRICS:
        raise KeyError(f"unknown metric '{metric}'")
    sel = (lambda m: m[roi]) if roi is not None else (lambda m: m.ravel())

    if metric in ("MSE", "MAE", "ΔE"):
        fn = {"MSE": metric_mse_map, "MAE": metric_mae_map, "ΔE": metric_deltaE_map}[metric]
        return float(aggregate_map(sel(fn(exp, stud)), aggregation or "mean")), tols
    if metric in ("RMSE", "PSNR"):
        mse = float(np.mean(sel(metric_mse_map(exp, stud))))
        if metric == "RMSE":
            return math.sqrt(mse), tols
        return (float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))), tols
    if metric == "SSIM":
        if roi is None:
            return metric_ssim(exp, stud), tols
        a = cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(stud, cv2.COLOR_BGR2GRAY)
        _, smap = sk_ssim(a, b, data_range=255, full=True)
        return float(np.mean(smap[roi])), tols
    if metric == "NCC":
        a = sel(cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY).astype(float))
        b = sel(cv2.cvtColor(stud, cv2.COLOR_BGR2GRAY).astype(float))
        a = a - a.mean(); b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return (0.0 if denom == 0 else float((a @ b) / denom)), tols
    if metric == "IoU":
        A = sel(cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY) > 127)
        B = sel(cv2.cvtColor(stud, cv2.COLOR_BGR2GRAY) > 127)
        uni = np.logical_or(A, B).sum()
        return (1.0 if uni == 0 else float(np.logical_and(A, B).sum() / uni)), tols
    if metric == "N_DIFF_PIXELS":
        thr, logic, use_gray, tol_clean = parse_ndiff_params(tols, notes)
        if use_gray:
            diff = np.abs(cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          - cv2.cvtColor(stud, cv2.COLOR_BGR2GRAY).astype(np.float32))
            bad = diff >= thr
        else:
            ad = np.abs(exp.astype(np.float32) - stud.astype(np.float32)) >= thr
            bad = ad.all(axis=2) if logic == "all" else ad.any(axis=2)
        return float(sel(bad).sum()), tol_clean
    if metric == "EXACT":
        a = exp[roi] if roi is not None else exp
        b = stud[roi] if roi is not None else stud
        return (1.0 if np.array_equal(a, b) else 0.0), tols
    # Shape/topology metrics: zero out pixels outside the ROI.
    if roi is not None:
        exp = exp * roi[:, :, None].astype(exp.dtype)
        stud = stud * roi[:, :, None].astype(stud.dtype)
    return float(METRICS[metric](exp, stud)), tols

# ------------------ Mask helpers ------------------
def overlay_mask_on_image(img_bgr, mask_gray, alpha=0.35):
    if img_bgr is None or mask_gray is None:
        return None
    h,w = img_bgr.shape[:2]
    if mask_gray.shape[:2] != (h,w):
        mask_gray = cv2.resize(mask_gray, (w,h), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    over = img_rgb.copy()
    red = np.zeros_like(over); red[:,:,0] = 255
    m = (mask_gray > 127)
    over[m] = np.clip((1.0 - alpha)*over[m] + alpha*red[m], 0, 255).astype(np.uint8)
    return over

def ensure_mask_in_notes(table, mask_path):
    """Add the global mask to image rules that have no mask of their own.
    A rule's own mask= is never overwritten (e.g. separate CL-ring / BG-ring masks)."""
    if not mask_path:
        return
    for r in range(table.rowCount()):
        out_item = table.item(r,0)
        if out_item and out_item.text().strip().lower() == "cout":
            continue
        item = table.item(r,6)
        txt = (item.text() if item else "").strip()
        if extract_mask_from_notes(txt):
            continue
        txt = f"{txt},mask={mask_path}" if txt else f"mask={mask_path}"
        if item:
            item.setText(txt)
        else:
            table.setItem(r,6,QTableWidgetItem(txt))

def extract_mask_from_notes(notes):
    if not isinstance(notes,str): return ""
    m = re.search(r"mask\s*=\s*([^,;]+)", notes, re.I)
    return m.group(1).strip() if m else ""

def resolve_path(path, base_dir=""):
    """Resolve a (possibly relative) path from a rule. Relative paths are tried
    against the folder of the imported rule file first, then the working
    directory. Returns '' if the file does not exist."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path if os.path.exists(path) else ""
    for root in ([base_dir] if base_dir else []) + [os.getcwd()]:
        p = os.path.normpath(os.path.join(root, path))
        if os.path.exists(p):
            return p
    return ""

def load_roi(mask_path, shape_hw):
    """Load a binary ROI mask (white = included) at the given (h, w).
    Raises if the file cannot be read, so a rule never runs silently unmasked."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {mask_path}")
    if m.shape[:2] != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    roi = m > 127
    if not roi.any():
        raise ValueError(f"mask {mask_path} is empty")
    return roi

# ------------------ UI: rectangle ROI (restored from commit 97a693e) ------------------
class RectRoiDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw ROI (drag to draw a rectangle)")
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise RuntimeError("Failed to load image for ROI drawing.")
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img_rgb.shape[:2]
        self.label = QLabel(); self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.pix = QPixmap.fromImage(QImage(self.img_rgb.data, self.w, self.h, self.w*3, QImage.Format_RGB888))
        self.view_w = min(900, self.w); self.view_h = int(self.h * (self.view_w/self.w))
        self.label.setPixmap(self.pix.scaled(self.view_w, self.view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Fix: the label is exactly the size of the shown image, so mouse
        # coordinates map 1:1 onto the preview.
        self.label.setFixedSize(self.label.pixmap().size())
        self.start = None; self.end = None; self.drawing = False
        lay = QVBoxLayout(self); lay.addWidget(self.label)
        btns = QHBoxLayout()
        bclr = QPushButton("Clear"); bok = QPushButton("Use ROI"); bcancel = QPushButton("Cancel")
        bclr.clicked.connect(self.clear); bok.clicked.connect(self.accept); bcancel.clicked.connect(self.reject)
        btns.addWidget(bclr); btns.addWidget(bok); btns.addWidget(bcancel); lay.addLayout(btns)
        self.label.mousePressEvent = self.on_press
        self.label.mouseMoveEvent = self.on_move
        self.label.mouseReleaseEvent = self.on_release
        self.overlay = QPixmap(self.label.pixmap())
        self.scale_x = self.w / self.label.pixmap().width()
        self.scale_y = self.h / self.label.pixmap().height()

    def clear(self):
        self.start = self.end = None
        self.label.setPixmap(self.pix.scaled(self.view_w, self.view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.overlay = QPixmap(self.label.pixmap())

    def on_press(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self.drawing = True
            self.start = ev.position().toPoint()
            self.end = self.start
            self.redraw()

    def on_move(self, ev):
        if self.drawing:
            self.end = ev.position().toPoint()
            self.redraw()

    def on_release(self, ev):
        if self.drawing:
            self.end = ev.position().toPoint()
            self.drawing = False
            self.redraw()

    def redraw(self):
        self.overlay = QPixmap(self.pix.scaled(self.view_w, self.view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        painter = QPainter(self.overlay)
        painter.setPen(QPen(QColor(255,0,0), 2, Qt.SolidLine))
        if self.start and self.end:
            painter.drawRect(QRect(self.start, self.end).normalized())
        painter.end()
        self.label.setPixmap(self.overlay)

    def rect_in_image(self):
        """Rectangle in full-resolution image coordinates (x1, y1, x2, y2)."""
        if not (self.start and self.end):
            return None
        r = QRect(self.start, self.end).normalized()
        # Fix: scale preview coordinates back to the original image size.
        x1 = int(np.clip(round(r.left() * self.scale_x), 0, self.w - 1))
        y1 = int(np.clip(round(r.top() * self.scale_y), 0, self.h - 1))
        x2 = int(np.clip(round(r.right() * self.scale_x), 0, self.w - 1))
        y2 = int(np.clip(round(r.bottom() * self.scale_y), 0, self.h - 1))
        return x1, y1, x2, y2

    def get_mask_path(self):
        rect = self.rect_in_image()
        if rect is None:
            return None
        x1, y1, x2, y2 = rect
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # Fix: save the mask where the instructor chooses (default: next to the
        # expected image) instead of a temp file that is overwritten each time.
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        default = os.path.join(os.path.dirname(self.image_path), f"{stem}_roi.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI mask", default, "PNG (*.png)")
        if not path:
            return None
        cv2.imwrite(path, mask)
        return path

# ------------------ Worker ------------------
class BatchWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(list)

    def __init__(self, student_dir, expected_dir, expected_image_path, rules, mask_path=None, rules_base_dir=""):
        super().__init__()
        self.student_dir = student_dir
        self.expected_dir = expected_dir
        self.expected_image_path = expected_image_path
        self.rules = rules
        self.mask_path = mask_path
        self.rules_base_dir = rules_base_dir  # folder of the imported rule file

        self._exp_index_exact = {}
        self._exp_cache_fuzzy = {}
        self._stud_cache = {}

    def _index_expected(self):
        if not self.expected_dir or self._exp_index_exact:
            return
        for root, _, files in os.walk(self.expected_dir):
            for fn in files:
                self._exp_index_exact.setdefault(fn.lower(), os.path.join(root, fn))

    def _find_in_expected(self, key_name: str):
        """Return (path, status). status: 'exact', 'stem', 'missing' or
        'AMBIGUOUS: a | b'. Ambiguous matches return no path (fail fast)."""
        if not key_name:
            return "", "missing"
        name_lower = key_name.lower()
        if name_lower in self._exp_index_exact:
            return self._exp_index_exact[name_lower], "exact"
        if name_lower in self._exp_cache_fuzzy:
            return self._exp_cache_fuzzy[name_lower]
        stem = self._strip_stud_suffix(name_lower).lower()
        cands = sorted(full for fn_lower, full in self._exp_index_exact.items()
                       if fn_lower.endswith(IMG_EXTS)
                       and self._strip_stud_suffix(fn_lower).lower() == stem)
        res = self._resolve_candidates(cands)
        self._exp_cache_fuzzy[name_lower] = res
        return res

    def _find_student_file(self, base_dir: str, target_name: str):
        """Return (path, status). Exact filename first, then exact stem match
        after stripping the student suffix. More than one candidate is
        reported as AMBIGUOUS and no file is used (fail fast)."""
        key = (base_dir, (target_name or "").lower())
        if key in self._stud_cache:
            return self._stud_cache[key]
        target_lower = key[1]
        exact, stem_hits = [], []
        stem = self._strip_stud_suffix(target_lower).lower() if target_lower else ""
        for root, _, files in os.walk(base_dir):
            for fn in files:
                fl = fn.lower()
                if fl == target_lower:
                    exact.append(os.path.join(root, fn))
                elif stem and fl.endswith(IMG_EXTS) and self._strip_stud_suffix(fl).lower() == stem:
                    stem_hits.append(os.path.join(root, fn))
        if exact:
            res = self._resolve_candidates(sorted(exact), "exact")
        else:
            res = self._resolve_candidates(sorted(stem_hits))
        self._stud_cache[key] = res
        return res

    @staticmethod
    def _resolve_candidates(cands, single_status="stem"):
        if len(cands) == 1:
            return cands[0], single_status
        if not cands:
            return "", "missing"
        return "", "AMBIGUOUS: " + " | ".join(os.path.basename(c) for c in cands)

    def _strip_stud_suffix(self, name: str) -> str:
        base = os.path.splitext(os.path.basename(name))[0]
        return re.sub(r'([_-])(stud|student)$', '', base, flags=re.IGNORECASE)

    def _safe_number(self, x):
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return 0.0
            return xf
        except Exception:
            return 0.0

    def run(self):
        self._index_expected()

        subs = [d for d in sorted(os.listdir(self.student_dir))
                if os.path.isdir(os.path.join(self.student_dir, d))]
        if not subs:
            subs = [""]

        explicit_expected = cv2.imread(self.expected_image_path) if self.expected_image_path else None

        results = []
        steps = max(1, len(subs) * max(1, len(self.rules)))
        step_done = 0

        for sub in subs:
            base = os.path.join(self.student_dir, sub) if sub else self.student_dir
            student_label = sub or os.path.basename(self.student_dir)

            rowres = {"student": student_label}
            total_score = 0.0

            # remember last resolved student path to support blank Output
            last_stud_path = None

            for ri, rule in enumerate(self.rules, 1):
                output = (rule.get("output") or "").strip()
                metric = (rule.get("metric") or "").strip()
                max_score = rule.get("max_score", 1.0)
                tols = rule.get("tolerances", "") or ""
                mode = rule.get("mode", "bucket") or "bucket"
                notes = rule.get("notes", "") or ""
                aggregation = rule.get("aggregation", "mean") or "mean"
                # Columns are keyed per rule so two rules with the same metric
                # (e.g. two N_DIFF_PIXELS rules on different ROIs) do not overwrite each other.
                rk = f"R{ri}_{metric}"

                if output.lower() == "cout":
                    txt = read_console_text(base, notes)
                    expected_text = expected_cout_text(tols, notes)
                    comp_mode = (metric or "exact").lower()
                    val = console_compare(txt, expected_text, comp_mode)
                    score = float(max_score) * (1.0 if val >= 1.0 else 0.0)
                    rowres[f"{rk}_val"] = val
                    rowres[f"{rk}_score"] = score
                    total_score += score
                    step_done += 1
                    self.progress.emit(int(step_done * 100 / steps), student_label)
                    continue

                # A rule's own mask= wins; otherwise the mask the instructor loaded
                # explicitly (never one taken from another rule).
                mask_spec = extract_mask_from_notes(notes) or (self.mask_path or "")
                mask_path_to_use = resolve_path(mask_spec, self.rules_base_dir) if mask_spec else ""
                rowres[f"{rk}_mask"] = mask_path_to_use or (f"MISSING: {mask_spec}" if mask_spec else "")

                # student file: use specified output; if blank, reuse last
                if output:
                    stud_path, stud_status = self._find_student_file(base, output)
                else:
                    stud_path, stud_status = (last_stud_path or ""), "reused"
                if stud_path:
                    last_stud_path = stud_path
                rowres[f"{rk}_student_path"] = stud_path or stud_status

                # expected image
                exp_img = None
                expected_path, exp_status = "", "missing"
                if explicit_expected is not None:
                    exp_img = explicit_expected
                    expected_path, exp_status = self.expected_image_path, "explicit"
                else:
                    if (output or stud_path) and self.expected_dir:
                        probe_name = output or os.path.basename(stud_path)
                        expected_path = map_expected_from_student(probe_name, self.expected_dir)
                        if expected_path:
                            exp_status = "exact"
                    if not expected_path and self.expected_dir and (output or stud_path):
                        probe = output or (os.path.basename(stud_path) if stud_path else "")
                        expected_path, exp_status = self._find_in_expected(probe)
                    if expected_path:
                        exp_img = cv2.imread(expected_path)
                rowres[f"{rk}_expected_path"] = expected_path or exp_status

                # SIZE fallback: accept WxH from notes/tols if no expected image
                if exp_img is None and metric == "SIZE":
                    ew, eh = parse_expected_size_any(notes, tols)
                    if ew and eh:
                        exp_img = np.zeros((eh, ew, 3), dtype=np.uint8)

                if exp_img is None:
                    rowres[f"{rk}_val"] = ("AMBIGUOUS expected" if exp_status.startswith("AMBIGUOUS")
                                           else "MISSING expected")
                    rowres[f"{rk}_score"] = 0.0
                    step_done += 1
                    self.progress.emit(int(step_done * 100 / steps), student_label)
                    continue

                stud_img = cv2.imread(stud_path) if stud_path else None
                if stud_img is None:
                    rowres[f"{rk}_val"] = ("AMBIGUOUS student" if stud_status.startswith("AMBIGUOUS")
                                           else "MISSING student")
                    rowres[f"{rk}_score"] = 0.0
                    step_done += 1
                    self.progress.emit(int(step_done * 100 / steps), student_label)
                    continue

                # Resize the student image to the expected resolution (INTER_AREA)
                # before masking, so a mask never meets two different sizes.
                # SIZE keeps the original dimensions. The resized flag is reported.
                resized = metric != "SIZE" and stud_img.shape[:2] != exp_img.shape[:2]
                if resized:
                    stud_img = cv2.resize(stud_img, (exp_img.shape[1], exp_img.shape[0]),
                                          interpolation=cv2.INTER_AREA)
                rowres[f"{rk}_resized"] = int(resized)

                err = None
                try:
                    roi = None
                    if mask_spec and metric != "SIZE":
                        if not mask_path_to_use:
                            raise FileNotFoundError(f"mask not found: {mask_spec}")
                        roi = load_roi(mask_path_to_use, exp_img.shape[:2])
                    val, tols = compute_metric(metric, exp_img, stud_img, roi,
                                               aggregation, tols, notes)
                    val = float(val)
                    # PSNR is +inf for identical images (a perfect result); any
                    # other NaN/inf is an error, never a silent "perfect" 0.0.
                    if math.isnan(val) or (math.isinf(val) and metric != "PSNR"):
                        raise ValueError("non-finite metric value")
                except Exception as e:
                    err = f"ERROR: {type(e).__name__}: {e}"

                if err is not None:
                    rowres[f"{rk}_val"] = err
                    rowres[f"{rk}_score"] = 0.0
                    step_done += 1
                    self.progress.emit(int(step_done * 100 / steps), student_label)
                    continue

                score = compute_score(metric, val, max_score, tols, mode)
                try:
                    if math.isnan(score) or math.isinf(score):
                        score = 0.0
                except Exception:
                    score = 0.0

                rowres[f"{rk}_val"] = val
                rowres[f"{rk}_score"] = score
                total_score += float(score)

                step_done += 1
                self.progress.emit(int(step_done * 100 / steps), student_label)

            rowres["total_score"] = total_score
            results.append(rowres)

        self.finished.emit(results)

# ------------------ Main Window ------------------
class ImageGraderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Assignment Grader (mask overlay + saved refs)")
        self.setGeometry(60,60,1400,780)
        self.expected_image_path=""; self.student_image_path=""; self.expected_dir=""; self.student_dir=""; self.mask_path=""
        self.rules_base_dir=""  # folder of the last imported rule file; relative mask paths resolve from here
        w=QWidget(); v=QVBoxLayout(w)
        top=QHBoxLayout()
        btnE=QPushButton("Load Expected Image"); btnE.clicked.connect(self.on_load_expected_image)
        btnS=QPushButton("Load Student Image"); btnS.clicked.connect(self.on_load_student_image)
        btnED=QPushButton("Select Expected Dir"); btnED.clicked.connect(self.on_pick_expected_dir)
        btnSD=QPushButton("Select Student Dir"); btnSD.clicked.connect(self.on_pick_student_dir)
        self.student_picker = QComboBox(); self.student_picker.setMinimumWidth(220)
        btnRefresh = QPushButton("Refresh Students"); btnRefresh.clicked.connect(self.refresh_student_list)
        btnMask=QPushButton("Load Mask / ROI"); btnMask.clicked.connect(self.on_load_mask)
        btnClear=QPushButton("Clear Images"); btnClear.clicked.connect(self.on_clear_images)
        btnReset=QPushButton("Reset All"); btnReset.clicked.connect(self.on_reset_all)
        self.mask_info = QLabel("Mask: (none)")
        for wdg in [btnE,btnS,btnED,btnSD, QLabel("Current Student:"), self.student_picker, btnRefresh, btnMask, btnClear, btnReset, self.mask_info]:
            top.addWidget(wdg)
        v.addLayout(top)
        split=QSplitter(Qt.Horizontal)
        self.lbl_expected=QLabel("Expected"); self.lbl_expected.setAlignment(Qt.AlignCenter)
        self.lbl_student=QLabel("Student"); self.lbl_student.setAlignment(Qt.AlignCenter)
        split.addWidget(self.lbl_expected); split.addWidget(self.lbl_student); split.setSizes([700,700]); v.addWidget(split)
        self.table=QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels(["Output (student filename)","Metric","MaxScore","Tolerance(s)","Mode","Aggregation","Notes","Result"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.table.setEditTriggers(QTableWidget.AllEditTriggers)
        v.addWidget(self.table)
        bottom=QHBoxLayout()
        btnAdd=QPushButton("Add Rule (popup)"); btnAdd.clicked.connect(self.on_add_rule_popup)
        btnDel=QPushButton("Delete Rule"); btnDel.clicked.connect(self.on_delete_rule)
        btnSingle=QPushButton("Run Single Eval"); btnSingle.clicked.connect(self.on_run_single)
        btnBatch=QPushButton("Batch Evaluate (student dir)"); btnBatch.clicked.connect(self.on_batch_eval)
        btnImp=QPushButton("Import Rules (JSON)"); btnImp.clicked.connect(self.on_import_json)
        btnExp=QPushButton("Export Rules (JSON)"); btnExp.clicked.connect(self.on_export_rules)
        for wdg in [btnAdd,btnDel,btnSingle,btnBatch,btnImp,btnExp]: bottom.addWidget(wdg)
        v.addLayout(bottom)
        self.rules_text=QTextEdit(); self.rules_text.setPlaceholderText("Rules text (JSON list).")
        v.addWidget(self.rules_text)
        util=QHBoxLayout()
        btnRef=QPushButton("Refresh Text from Table"); btnRef.clicked.connect(self.on_refresh_rules_text)
        btnApply=QPushButton("Apply Text to Table"); btnApply.clicked.connect(self.on_apply_rules_text)
        btnScan=QPushButton("Scan Assignment Folder"); btnScan.clicked.connect(self.on_scan_assignment)
        for wdg in [btnRef,btnApply,btnScan]: util.addWidget(wdg)
        v.addLayout(util)
        self.setCentralWidget(w)

    def _update_previews(self):
        if self.expected_image_path:
            self._show_image(self.expected_image_path, self.lbl_expected, with_mask=True)
        else:
            self.lbl_expected.setPixmap(QPixmap()); self.lbl_expected.setText("Expected")
        if self.student_image_path:
            self._show_image(self.student_image_path, self.lbl_student, with_mask=True)
        else:
            self.lbl_student.setPixmap(QPixmap()); self.lbl_student.setText("Student")

    def _show_image(self, path, label, max_size=420, with_mask=False):
        img=cv2.imread(path)
        if img is None:
            label.setText("Failed to load"); return None
        if with_mask and self.mask_path and os.path.exists(self.mask_path):
            m = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                rgb = overlay_mask_on_image(img, m)
                if rgb is not None:
                    h,w=rgb.shape[:2]
                    q=QImage(rgb.data, w,h, 3*w, QImage.Format_RGB888)
                    label.setPixmap(QPixmap.fromImage(q).scaled(max_size,max_size,Qt.KeepAspectRatio))
                    return img
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h,w,ch=img_rgb.shape
        q=QImage(img_rgb.data, w,h, ch*w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q).scaled(max_size,max_size,Qt.KeepAspectRatio))
        return img

    def on_pick_expected_dir(self):
        d=QFileDialog.getExistingDirectory(self,"Select Expected Directory")
        if d: self.expected_dir=d; QMessageBox.information(self,"Selected",f"Expected directory set to:\n{d}")

    def on_pick_student_dir(self):
        d=QFileDialog.getExistingDirectory(self,"Select Student Directory (parent or single student)")
        if d:
            self.student_dir=d; self.refresh_student_list()
            QMessageBox.information(self,"Selected",f"Student directory set to:\n{d}")

    def refresh_student_list(self):
        self.student_picker.clear()
        subs=[]
        if self.student_dir and os.path.isdir(self.student_dir):
            subs = sorted([n for n in os.listdir(self.student_dir) if os.path.isdir(os.path.join(self.student_dir, n))])
        if subs: self.student_picker.addItems(subs)
        else: self.student_picker.addItem("(no subfolders)")

    def on_load_expected_image(self):
        p,_=QFileDialog.getOpenFileName(self,"Open expected image","","Images (*.png *.jpg *.jpeg)")
        if p:
            self.expected_image_path=p
            self._update_previews()

    def on_load_student_image(self):
        p,_=QFileDialog.getOpenFileName(self,"Open student image","","Images (*.png *.jpg *.jpeg)")
        if p:
            self.student_image_path=p
            self._update_previews()

    def on_load_mask(self):
        btn = QMessageBox.question(self, "Mask / ROI",
                                   "Do you want to draw a rectangle ROI?\n(Choose 'No' to load a mask image from file.)",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
        if btn == QMessageBox.Cancel:
            return
        if btn == QMessageBox.Yes:
            if not self.expected_image_path:
                QMessageBox.warning(self,"Missing expected image","Please load an expected image first to draw an ROI.")
                return
            try:
                dlg = RectRoiDialog(self.expected_image_path, self)  # noqa: F821
                if dlg.exec() == QDialog.Accepted:
                    mp = dlg.get_mask_path()
                    if mp:
                        self.mask_path = mp
                        self.mask_info.setText(f"Mask: {os.path.basename(mp)}")
                        ensure_mask_in_notes(self.table, self.mask_path)
                        QMessageBox.information(self,"ROI","Rectangle ROI captured and applied.")
                        self._update_previews()
                        return
                    else:
                        QMessageBox.warning(self,"ROI","No rectangle was drawn.")
                        return
            except Exception as e:
                QMessageBox.warning(self,"ROI error",f"Failed to draw ROI: {e}")
                return
        p,_=QFileDialog.getOpenFileName(self,"Open mask image (binary)","","Images (*.png *.jpg *.jpeg)")
        if p:
            self.mask_path=p
            self.mask_info.setText(f"Mask: {os.path.basename(p)}")
            ensure_mask_in_notes(self.table, self.mask_path)
            QMessageBox.information(self,"Mask",f"Mask loaded: {p}")
            self._update_previews()

    def on_clear_images(self):
        self.expected_image_path=""; self.student_image_path=""
        self.lbl_expected.setPixmap(QPixmap()); self.lbl_expected.setText("Expected")
        self.lbl_student.setPixmap(QPixmap()); self.lbl_student.setText("Student")

    def on_reset_all(self):
        self.on_clear_images(); self.expected_dir=""; self.student_dir=""; self.mask_path=""; self.rules_base_dir=""
        self.mask_info.setText("Mask: (none)")
        self.table.setRowCount(0); self.student_picker.clear(); self.student_picker.addItem("(no subfolders)")

    def on_add_rule_popup(self):
        autofill = os.path.basename(self.student_image_path) if self.student_image_path else ""
        dlg=RulePopup(self, autofill)
        if dlg.exec():
            r=dlg.get_rule(); row=self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem(r["output"])); self.table.setItem(row,1,QTableWidgetItem(r["metric"]))
            self.table.setItem(row,2,QTableWidgetItem(str(r["max_score"]))); self.table.setItem(row,3,QTableWidgetItem(r["tolerances"]))
            self.table.setItem(row,4,QTableWidgetItem(r["mode"])); self.table.setItem(row,5,QTableWidgetItem(r["aggregation"]))
            notes = r["notes"]
            if self.mask_path and f"mask={self.mask_path}" not in notes:
                notes = (notes + "," if notes else "") + f"mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem("Not run"))

    def on_delete_rule(self):
        r=self.table.currentRow()
        if r>=0: self.table.removeRow(r)

    def _mask_spec(self, notes_text):
        """The rule's own mask, else the explicitly loaded global mask."""
        return extract_mask_from_notes(notes_text) or (self.mask_path or "")

    def on_run_single(self):
        has_cout = any((self.table.item(r,0) and self.table.item(r,0).text().strip().lower()=="cout") for r in range(self.table.rowCount()))
        exp=None; stud=None
        if not has_cout:
            if not self.expected_image_path or not self.student_image_path:
                QMessageBox.warning(self,"Missing","Load both expected and student image to run single evaluation."); return
            exp=cv2.imread(self.expected_image_path); stud=cv2.imread(self.student_image_path)
            if exp is None or stud is None: QMessageBox.warning(self,"Error","Failed to load images."); return

        res=[]
        for row in range(self.table.rowCount()):
            out_item=self.table.item(row,0); met_item=self.table.item(row,1); max_item=self.table.item(row,2)
            tol_item=self.table.item(row,3); mode_item=self.table.item(row,4); agg_item=self.table.item(row,5); notes_item=self.table.item(row,6)
            if not met_item: continue
            out = out_item.text().strip() if out_item else ""; metric=met_item.text().strip()
            max_score=float(max_item.text()) if max_item else 1.0; tol=tol_item.text() if tol_item else ""; mode=mode_item.text() if mode_item else "bucket"
            agg=agg_item.text().strip() if agg_item else ""; notes=notes_item.text().strip() if notes_item else ""

            if out.lower()=="cout":
                if self.student_dir:
                    sel=self.student_picker.currentText().strip() if self.student_picker.count()>0 else ""
                    if sel and sel!="(no subfolders)" and os.path.isdir(os.path.join(self.student_dir, sel)):
                        base_dir=os.path.join(self.student_dir, sel)
                    else:
                        base_dir=self.student_dir
                elif self.student_image_path:
                    base_dir=os.path.dirname(self.student_image_path)
                else:
                    base_dir=""
                if not base_dir: self.table.setItem(row,7,QTableWidgetItem("No student dir/image for cout")); continue
                txt=read_console_text(base_dir, notes)
                exp_txt = expected_cout_text(tol, notes)
                val = console_compare(txt, exp_txt, metric.lower() if metric else "exact")
                score = max_score * (1.0 if val>=1.0 else 0.0)
                self.table.setItem(row,7,QTableWidgetItem(f"cout: {val:.1f} | score {score:.2f}"))
                res.append(f"COUT ({metric}) [{os.path.basename(base_dir)}]: {val:.1f} -> {score:.2f}")
                continue

            A=exp.copy() if exp is not None else None; B=stud.copy() if stud is not None else None
            if A is None or B is None: self.table.setItem(row,7,QTableWidgetItem("No images loaded")); continue
            if metric != "SIZE" and B.shape[:2] != A.shape[:2]:
                B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
            # Same evaluation path as batch mode (compute_metric).
            try:
                roi = None
                spec = self._mask_spec(notes)
                if spec and metric != "SIZE":
                    mpath = resolve_path(spec, self.rules_base_dir)
                    if not mpath:
                        raise FileNotFoundError(f"mask not found: {spec}")
                    roi = load_roi(mpath, A.shape[:2])
                val, tol = compute_metric(metric, A, B, roi, agg, tol, notes)
                if math.isnan(val) or (math.isinf(val) and metric != "PSNR"):
                    raise ValueError("non-finite metric value")
            except Exception as e:
                msg = f"ERROR: {type(e).__name__}: {e}"
                self.table.setItem(row,7,QTableWidgetItem(msg)); res.append(f"{metric}: {msg}")
                continue
            score=compute_score(metric, val, max_score, tol, mode)
            self.table.setItem(row,7,QTableWidgetItem(f"Value {val:.4f} | Score {score:.2f}"))
            res.append(f"{metric}: {val:.4f} -> {score:.2f}")
        QMessageBox.information(self,"Single Evaluation Results","\n".join(res))
        self._update_previews()

    def on_batch_eval(self):
        if not self.student_dir: QMessageBox.warning(self,"Missing","Please select a student directory."); return
        rules=[]
        for row in range(self.table.rowCount()):
            rules.append({
                "output": self.table.item(row,0).text() if self.table.item(row,0) else "",
                "metric": self.table.item(row,1).text() if self.table.item(row,1) else "",
                "max_score": float(self.table.item(row,2).text()) if self.table.item(row,2) else 1.0,
                "tolerances": self.table.item(row,3).text() if self.table.item(row,3) else "",
                "mode": self.table.item(row,4).text() if self.table.item(row,4) else "bucket",
                "aggregation": self.table.item(row,5).text() if self.table.item(row,5) else "mean",
                "notes": self.table.item(row,6).text() if self.table.item(row,6) else ""
            })
        if not rules: QMessageBox.warning(self,"No rules","Add at least one rule to run batch evaluation."); return
        save,_=QFileDialog.getSaveFileName(self,"Save CSV results","","CSV Files (*.csv)")
        if not save: return
        self.progress=QProgressDialog("Batch running...","Cancel",0,100,self); self.progress.setWindowModality(Qt.WindowModal); self.progress.setAutoClose(False); self.progress.setValue(0)
        self.worker=BatchWorker(self.student_dir, self.expected_dir, self.expected_image_path or "", rules, self.mask_path or None, self.rules_base_dir)
        self.worker.progress.connect(self.on_worker_progress); self.worker.finished.connect(lambda res: self.on_batch_finished(res, save))
        self.worker.start(); self.progress.exec_()

    def on_worker_progress(self, percent, fname):
        self.progress.setValue(percent); self.progress.setLabelText(f"Processing: {fname}"); QApplication.processEvents()

    def on_batch_finished(self, results, save_path):
        try: self.progress.setValue(100); self.progress.close()
        except: pass
        if results:
            # Keep columns in rule order (R1..Rn), total_score last.
            keys=list(dict.fromkeys(k for r in results for k in r.keys()))
            keys=["student"]+[k for k in keys if k not in ("student","total_score")]+["total_score"]
            try:
                with open(save_path,"w",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in results]
                QMessageBox.information(self,"Batch Complete",f"Batch finished. Results saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.warning(self,"Save error",f"Failed to save CSV: {e}")
        else:
            QMessageBox.information(self,"Batch Complete","No results generated.")

    def on_import_json(self):
        p,_=QFileDialog.getOpenFileName(self,"Load rules JSON","","JSON Files (*.json)")
        if not p: return
        try:
            with open(p,"r",encoding="utf-8") as f: rules=json.load(f)
            self.table.setRowCount(0)
            self.rules_base_dir = os.path.dirname(os.path.abspath(p))
            for r in rules:
                notes, tol_text = _cout_fields(r)
                row=self.table.rowCount(); self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(r.get("output",""))); self.table.setItem(row,1,QTableWidgetItem(r.get("metric","")))
                self.table.setItem(row,2,QTableWidgetItem(str(r.get("max_score",1.0)))); self.table.setItem(row,3,QTableWidgetItem(tol_text))
                self.table.setItem(row,4,QTableWidgetItem(r.get("mode","bucket"))); self.table.setItem(row,5,QTableWidgetItem(r.get("aggregation","mean")))
                self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem(r.get("result","")))
            QMessageBox.information(self,"Loaded",f"Loaded rules from {p}")
        except Exception as e:
            QMessageBox.warning(self,"Error",f"Failed to load JSON: {e}")

    def on_export_rules(self):
        p,_=QFileDialog.getSaveFileName(self,"Export rules JSON","","JSON Files (*.json)")
        if not p: return
        rules=[]
        for row in range(self.table.rowCount()):
            notes = self.table.item(row,6).text() if self.table.item(row,6) else ""
            nd = parse_notes_kv(notes)
            rules.append({
                "output": self.table.item(row,0).text() if self.table.item(row,0) else "",
                "metric": self.table.item(row,1).text() if self.table.item(row,1) else "",
                "max_score": float(self.table.item(row,2).text()) if self.table.item(row,2) else 1.0,
                "tolerances": self.table.item(row,3).text() if self.table.item(row,3) else "",
                "mode": self.table.item(row,4).text() if self.table.item(row,4) else "bucket",
                "aggregation": self.table.item(row,5).text() if self.table.item(row,5) else "mean",
                "notes": notes,
                "expected_cout": expected_cout_text(self.table.item(row,3).text() if self.table.item(row,3) else "", notes) if (self.table.item(row,0) and self.table.item(row,0).text().strip().lower()=="cout") else "",
                "result": self.table.item(row,7).text() if self.table.item(row,7) else ""
            })
        try:
            with open(p,"w",encoding="utf-8") as f: json.dump(rules,f,indent=2)
            QMessageBox.information(self,"Exported",f"Rules exported to {p}")
        except Exception as e:
            QMessageBox.warning(self,"Error",f"Failed to export rules: {e}")

    def on_refresh_rules_text(self):
        rules=[]
        for row in range(self.table.rowCount()):
            out=self.table.item(row,0); met=self.table.item(row,1); mx=self.table.item(row,2); tol=self.table.item(row,3); mode=self.table.item(row,4); agg=self.table.item(row,5); notes=self.table.item(row,6)
            if not (out and met and mx): continue
            notes_text = notes.text().strip() if notes else ""
            nd = parse_notes_kv(notes_text)
            rules.append({
                "output": out.text().strip(),"metric": met.text().strip(),
                "max_score": float(mx.text().strip() or 1.0),
                "tolerances": tol.text().strip() if tol else "",
                "mode": mode.text().strip() if mode else "bucket",
                "aggregation": agg.text().strip() if agg else "mean",
                "notes": notes_text,
                "expected_cout": expected_cout_text(tol.text() if tol else "", notes_text) if out.text().strip().lower()=="cout" else ""
            })
        self.rules_text.setPlainText(json.dumps(rules, indent=2))

    def on_apply_rules_text(self):
        try: rules=json.loads(self.rules_text.toPlainText())
        except Exception as e: QMessageBox.warning(self,"Invalid JSON",f"{e}"); return
        self.table.setRowCount(0)
        for r,rule in enumerate(rules):
            self.table.insertRow(r)
            self.table.setItem(r,0,QTableWidgetItem(rule.get("output",""))); self.table.setItem(r,1,QTableWidgetItem(rule.get("metric","")))
            notes, tol_text = _cout_fields(rule)
            self.table.setItem(r,2,QTableWidgetItem(str(rule.get("max_score",1.0)))); self.table.setItem(r,3,QTableWidgetItem(tol_text))
            self.table.setItem(r,4,QTableWidgetItem(rule.get("mode","bucket"))); self.table.setItem(r,5,QTableWidgetItem(rule.get("aggregation","mean")))
            self.table.setItem(r,6,QTableWidgetItem(notes)); self.table.setItem(r,7,QTableWidgetItem(""))

    def on_scan_assignment(self):
        d=QFileDialog.getExistingDirectory(self,"Select assignment folder")
        if not d: return
        outs=set(); console=False
        literal_pat = re.compile(r"['\"]([^'\"\n]+?(?:_stud(?:ent)?)[^'\"\n]*\.(?:png|jpg|jpeg))['\"]", re.I)
        concat_pat = re.compile(r"['\"]([A-Za-z0-9_\-\/\\]+)['\"]\s*\+\s*['\"](_stud(?:ent)?\.(?:png|jpg|jpeg))['\"]", re.I)
        join_pat = re.compile(r"os\.path\.join\([^)]*['\"]([^'\"\n]+_stud(?:ent)?\.(?:png|jpg|jpeg))['\"][^)]*\)", re.I)

        for root,_,files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(('.py','.cpp','.hpp','.h','.c','.java','.ipynb','.txt')):
                    try:
                        with open(os.path.join(root,fn),'r',encoding='utf-8',errors='ignore') as f: s=f.read()
                    except: continue
                    if ('print(' in s) or ('std::cout' in s) or ('cout <<' in s): console=True
                    for m in literal_pat.finditer(s):
                        outs.add(os.path.basename(m.group(1)))
                    for m in concat_pat.finditer(s):
                        base = m.group(1).rstrip("/\\"); suffix = m.group(2)
                        name = os.path.basename(base) + suffix
                        outs.add(os.path.basename(name))
                    for m in join_pat.finditer(s):
                        outs.add(os.path.basename(m.group(1)))

        row=self.table.rowCount()
        for o in sorted(outs):
            self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem(o)); self.table.setItem(row,1,QTableWidgetItem('EXACT'))
            self.table.setItem(row,2,QTableWidgetItem('1')); self.table.setItem(row,3,QTableWidgetItem(''))
            self.table.setItem(row,4,QTableWidgetItem('bucket')); self.table.setItem(row,5,QTableWidgetItem('mean'))
            notes = 'from scan'
            if self.mask_path:
                notes += f",mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem('')); row+=1
        if console:
            self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem('cout')); self.table.setItem(row,1,QTableWidgetItem('exact'))
            self.table.setItem(row,2,QTableWidgetItem('1')); self.table.setItem(row,3,QTableWidgetItem('')
            )
            self.table.setItem(row,4,QTableWidgetItem('bucket')); self.table.setItem(row,5,QTableWidgetItem(''))
            notes = 'file=stdout.txt'
            if self.mask_path:
                notes += f",mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem(''))

if __name__ == "__main__":
    app=QApplication(sys.argv); win=ImageGraderApp(); win.show(); sys.exit(app.exec())

