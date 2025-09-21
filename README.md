# the full Code
import sys, os, csv, json, math, re, tempfile
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QFormLayout, QLineEdit, QComboBox, QMessageBox,
    QSplitter, QProgressDialog, QSpinBox, QTextEdit
)

from skimage.metrics import structural_similarity as sk_ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from collections import Counter

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

def metric_count_objects(a, _b=None):
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    n, _ = cv2.connectedComponents(th)
    return max(0, int(n-1))

def metric_count_holes(a, _b=None):
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    n, _ = cv2.connectedComponents(inv)
    return max(0, int(n-1))

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

# NEW: number of different pixels above a threshold
def metric_n_diff_pixels(a, b, thr=10, logic='any', use_gray=False):
    """Count pixels whose difference exceeds a threshold."""
    a, b = _ensure_same_size(a, b)
    if use_gray:
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ga - gb)
        return float((diff > thr).sum())
    else:
        ad = np.abs(a.astype(np.float32) - b.astype(np.float32))
        if logic == 'all':
            mask = (ad[:,:,0] > thr) & (ad[:,:,1] > thr) & (ad[:,:,2] > thr)
        else:
            mask = (ad[:,:,0] > thr) | (ad[:,:,1] > thr) | (ad[:,:,2] > thr)
        return float(mask.sum())

# NEW: stricter contour hierarchy similarity (0..1)
def _holes_signature(gray):
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return []
    hier = hier[0]
    holes_per_outer = []
    for idx, h in enumerate(hier):
        if h[3] == -1:  # outer contour
            c = sum(1 for j, hh in enumerate(hier) if hh[3] == idx)
            holes_per_outer.append(c)
    holes_per_outer.sort()
    return holes_per_outer

def metric_contours_strict(a, b):
    """Return IoU-like similarity (0..1) of the histogram of 'holes per outer contour'."""
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

# --------- helpers for N_DIFF_PIXELS ---------
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

# ---------- console helpers ----------
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
    """Parse key=value pairs in Notes (comma/semicolon separated)."""
    out={}
    if not isinstance(notes,str): return out
    for m in re.finditer(r"(\w+)\s*=\s*([^,;]+)", notes):
        out[m.group(1).lower()] = m.group(2).strip()
    return out

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
    "Contours": metric_contour_hierarchy,
    "ContoursStrict": metric_contours_strict,     # NEW
    "EXACT": metric_exact,
    "N_DIFF_PIXELS": metric_n_diff_pixels         # NEW
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
    "Contours": "1",
    "ContoursStrict": "1,0.75,0.5",
    "EXACT": "1",
    "N_DIFF_PIXELS": "thr:10 | 100,1000,5000"
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

# ------------------ UI: Add Rule ------------------
class RulePopup(QDialog):
    def __init__(self, parent=None, autofill_output=""):
        super().__init__(parent)
        self.setWindowTitle("Add Rule")
        layout = QFormLayout(self)

        self.output_field = QLineEdit(autofill_output)
        self.metric_box = QComboBox(); self.metric_box.addItems(list(METRICS.keys()))
        self.agg_box = QComboBox(); self.agg_box.addItems(['mean','median','max','trim10'])
        self.output_field.textChanged.connect(self.on_output_change)
        self.max_score = QSpinBox(); self.max_score.setRange(0,100); self.max_score.setValue(1)
        self.tol_edit = QLineEdit()
        self.cout_expected_edit = QLineEdit()   # field for cout
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
            if self.metric_box.count()!=len(METRICS):
                self.metric_box.clear(); self.metric_box.addItems(list(METRICS.keys()))
        self.tol_edit.setVisible(not is_cout)
        self.cout_expected_edit.setVisible(is_cout)
        self.agg_box.setEnabled(False if is_cout else self.agg_box.isEnabled())

    def on_metric_change(self, metric):
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
            expected_cout = self.cout_expected_edit.text().strip()
            if expected_cout:
                prefix = f"exp_cout={expected_cout}"
                notes_text = f"{prefix},{notes_text}" if notes_text else prefix
        return {
            "output": out,
            "metric": self.metric_box.currentText(),
            "max_score": float(self.max_score.value()),
            "tolerances": "" if out.lower()=="cout" else self.tol_edit.text().strip(),
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

# ------------------ Mask helpers ------------------
def overlay_mask_on_image(img_bgr, mask_gray, alpha=0.35):
    """Return an RGB image with a red, semi-transparent mask overlay."""
    if img_bgr is None or mask_gray is None:
        return None
    h,w = img_bgr.shape[:2]
    if mask_gray.shape[:2] != (h,w):
        mask_gray = cv2.resize(mask_gray, (w,h), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    over = img_rgb.copy()
    red = np.zeros_like(over); red[:,:,0] = 255  # RGB red
    m = (mask_gray > 127)
    over[m] = np.clip((1.0 - alpha)*over[m] + alpha*red[m], 0, 255).astype(np.uint8)
    return over

def ensure_mask_in_notes(table, mask_path):
    """Guarantee 'mask=<path>' appears in Notes for all rows."""
    if not mask_path:
        return
    for r in range(table.rowCount()):
        item = table.item(r,6)
        txt = (item.text() if item else "").strip()
        if f"mask={mask_path}" in txt:
            continue
        # remove any old mask=... to avoid duplicates
        txt = re.sub(r"mask\s*=\s*[^,;]+", "", txt)
        txt = txt.strip(", ;")
        txt = f"{txt},mask={mask_path}" if txt else f"mask={mask_path}"
        if item:
            item.setText(txt)
        else:
            table.setItem(r,6,QTableWidgetItem(txt))

def extract_mask_from_notes(notes):
    if not isinstance(notes,str): return ""
    m = re.search(r"mask\s*=\s*([^,;]+)", notes, re.I)
    return m.group(1).strip() if m else ""

# ------------------ Worker ------------------
class BatchWorker(QThread):
    progress = Signal(int, str); finished = Signal(list)
    def __init__(self, student_dir, expected_dir, expected_image_path, rules, mask_path=None):
        super().__init__(); self.student_dir=student_dir; self.expected_dir=expected_dir
        self.expected_image_path=expected_image_path; self.rules=rules; self.mask_path=mask_path
    def run(self):
        subs = [f for f in sorted(os.listdir(self.student_dir)) if os.path.isdir(os.path.join(self.student_dir,f))]
        if not subs: subs=[""]
        total=max(1,len(subs)); results=[]; exp_img=None
        if self.expected_image_path: exp_img = cv2.imread(self.expected_image_path)
        base_mask = None
        if self.mask_path and os.path.exists(self.mask_path):
            base_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if base_mask is not None: base_mask=(base_mask>127).astype(np.uint8)
        for i, sub in enumerate(subs):
            base = os.path.join(self.student_dir, sub) if sub else self.student_dir
            rowres={"student": sub or os.path.basename(self.student_dir)}; total_score=0.0
            for rule in self.rules:
                output=rule.get("output",""); metric=rule.get("metric",""); max_score=rule.get("max_score",1.0)
                tols=rule.get("tolerances",""); mode=rule.get("mode","bucket"); notes=rule.get("notes","")
                aggregation=rule.get("aggregation","mean")
                # Row-level mask override
                mask_path_notes = extract_mask_from_notes(notes)
                mask_to_use = mask_path_notes if mask_path_notes else self.mask_path
                rowres["mask"] = mask_to_use or ""
                mask = None
                if mask_to_use and os.path.exists(mask_to_use):
                    mask = cv2.imread(mask_to_use, cv2.IMREAD_GRAYSCALE)
                    if mask is not None: mask=(mask>127).astype(np.uint8)

                if output.lower()=="cout":
                    txt = read_console_text(base, notes)
                    nd = parse_notes_kv(notes)
                    exp_txt = nd.get("exp_cout") or rule.get("expected_cout","") or tols or notes
                    val = console_compare(txt, exp_txt, metric.lower() if metric else "exact")
                    score = max_score * (1.0 if val>=1.0 else 0.0)
                else:
                    eimg = exp_img
                    simg = None
                    if output:
                        sp = os.path.join(base, output)
                        if os.path.exists(sp): simg = cv2.imread(sp)
                    if eimg is None or simg is None:
                        val=float('nan'); score=0.0
                    else:
                        A=eimg.copy(); B=simg.copy()
                        if mask is not None:
                            if A.shape[:2]!=mask.shape:
                                mres=cv2.resize(mask,(A.shape[1],A.shape[0]),interpolation=cv2.INTER_NEAREST)
                            else: mres=mask
                            A=A*mres[:,:,None]; B=B*mres[:,:,None]
                        func=METRICS.get(metric)
                        try:
                            if metric in {"Objects","Holes"}:
                                val=float(func(B,None))
                            elif metric in {"Contours","ContoursStrict"}:
                                val=float(func(A,B))
                            elif metric=="ΔE" and aggregation:
                                val=aggregate_map(metric_deltaE_map(A,B), aggregation)
                            elif metric=="MAE" and aggregation:
                                val=aggregate_map(metric_mae_map(A,B), aggregation)
                            elif metric=="MSE" and aggregation:
                                val=aggregate_map(metric_mse_map(A,B), aggregation)
                            elif metric=="N_DIFF_PIXELS":
                                thr, logic, use_gray, tol_clean = parse_ndiff_params(tols, notes)
                                val = float(metric_n_diff_pixels(A,B,thr=thr,logic=logic,use_gray=use_gray))
                                tols = tol_clean
                            else:
                                val=float(func(A,B)) if func else float('nan')
                        except Exception:
                            val=float('nan')
                        score=compute_score(metric, val, max_score, tols, mode)
                rowres[f"{metric}_val"]=val; rowres[f"{metric}_score"]=score; total_score+=float(score)
            rowres["total_score"]=total_score; results.append(rowres)
            self.progress.emit(int((i+1)/total*100), sub or os.path.basename(self.student_dir))
        self.finished.emit(results)

# ----- Interactive ROI rectangle drawing -----
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
        self.label = QLabel(); self.label.setAlignment(Qt.AlignCenter)
        self.pix = QPixmap.fromImage(QImage(self.img_rgb.data, self.w, self.h, self.w*3, QImage.Format_RGB888))
        self.view_w = min(900, self.w); self.view_h = int(self.h * (self.view_w/self.w))
        self.label.setPixmap(self.pix.scaled(self.view_w, self.view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        self.overlay = QPixmap(self.label.pixmap())
        painter = QPainter(self.overlay)
        painter.setPen(QPen(QColor(255,0,0), 2, Qt.SolidLine))
        if self.start and self.end:
            r = QRect(self.start, self.end).normalized()
            painter.drawRect(r)
        painter.end()
        self.label.setPixmap(self.overlay)

    def get_mask_path(self):
        if not (self.start and self.end):
            return None
        r = QRect(self.start, self.end).normalized()
        x1 = int(r.left()); y1 = int(r.top())
        x2 = int(r.right()); y2 = int(r.bottom())
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        tmp = os.path.join(tempfile.gettempdir(), "roi_mask_rect.png")
        cv2.imwrite(tmp, mask)
        return tmp

# ------------------ Main Window ------------------
class ImageGraderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Assignment Grader (mask overlay + saved refs)")
        self.setGeometry(60,60,1400,780)
        self.expected_image_path=""; self.student_image_path=""; self.expected_dir=""; self.student_dir=""; self.mask_path=""
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
        self.mask_info = QLabel("Mask: (none)")  # shows current mask basename
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
        btnExp=QPushButton("Export Rules (JSON/ugly)"); btnExp.clicked.connect(self.on_export_rules)
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

    # ---------- preview helpers ----------
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

    # ---------- top buttons ----------
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
                dlg = RectRoiDialog(self.expected_image_path, self)
                if dlg.exec() == QDialog.Accepted:
                    mp = dlg.get_mask_path()
                    if mp:
                        self.mask_path = mp
                        self.mask_info.setText(f"Mask: {os.path.basename(mp)}")
                        ensure_mask_in_notes(self.table, self.mask_path)  # save to refs
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
            ensure_mask_in_notes(self.table, self.mask_path)  # save to refs
            QMessageBox.information(self,"Mask",f"Mask loaded: {p}")
            self._update_previews()

    def on_clear_images(self):
        self.expected_image_path=""; self.student_image_path=""
        self.lbl_expected.setPixmap(QPixmap()); self.lbl_expected.setText("Expected")
        self.lbl_student.setPixmap(QPixmap()); self.lbl_student.setText("Student")

    def on_reset_all(self):
        self.on_clear_images(); self.expected_dir=""; self.student_dir=""; self.mask_path=""
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
            # Ensure new rows also carry mask reference if any
            notes = r["notes"]
            if self.mask_path and f"mask={self.mask_path}" not in notes:
                notes = (notes + "," if notes else "") + f"mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem("Not run"))

    def on_delete_rule(self):
        r=self.table.currentRow()
        if r>=0: self.table.removeRow(r)

    def _load_mask_from_notes_or_global(self, notes_text):
        mpath = extract_mask_from_notes(notes_text)
        if mpath and os.path.exists(mpath):
            return mpath
        return self.mask_path if (self.mask_path and os.path.exists(self.mask_path)) else ""

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
            # Resolve mask from notes or global
            mask_path = self._load_mask_from_notes_or_global(notes)
            mask=None
            if mask_path:
                mimg=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mimg is not None: mask=(mimg>127).astype(np.uint8)

            if out.lower()=="cout":
                # choose base dir
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
                nd = parse_notes_kv(notes)
                exp_txt = nd.get("exp_cout") or tol or notes
                val = console_compare(txt, exp_txt, metric.lower() if metric else "exact")
                score = max_score * (1.0 if val>=1.0 else 0.0)
                self.table.setItem(row,7,QTableWidgetItem(f"cout: {val:.1f} | score {score:.2f}"))
                res.append(f"COUT ({metric}) [{os.path.basename(base_dir)}]: {val:.1f} -> {score:.2f}")
                continue

            A=exp.copy() if exp is not None else None; B=stud.copy() if stud is not None else None
            if A is None or B is None: self.table.setItem(row,7,QTableWidgetItem("No images loaded")); continue
            if mask is not None:
                if A.shape[:2]!=mask.shape: mres=cv2.resize(mask,(A.shape[1],A.shape[0]),interpolation=cv2.INTER_NEAREST)
                else: mres=mask
                A=A*mres[:,:,None]; B=B*mres[:,:,None]
            func=METRICS.get(metric)
            if func is None: val=float('nan')
            else:
                try:
                    if metric in {"Objects","Holes"}: val=float(func(B,None))
                    elif metric in {"Contours","ContoursStrict"}: val=float(func(A,B))
                    elif metric=="ΔE" and agg: val=float(aggregate_map(metric_deltaE_map(A,B), agg))
                    elif metric=="MAE" and agg: val=float(aggregate_map(metric_mae_map(A,B), agg))
                    elif metric=="MSE" and agg: val=float(aggregate_map(metric_mse_map(A,B), agg))
                    elif metric=="N_DIFF_PIXELS":
                        thr, logic, use_gray, tol_clean = parse_ndiff_params(tol, notes)
                        val = float(metric_n_diff_pixels(A,B,thr=thr,logic=logic,use_gray=use_gray))
                        tol = tol_clean  # for scoring
                    else: val=float(func(A,B))
                except Exception: val=float('nan')
            score=compute_score(metric, val, max_score, tol, mode)
            self.table.setItem(row,7,QTableWidgetItem(f"{val:.4f} | score {score:.2f}"))
            res.append(f"{metric}: {val:.4f} -> {score:.2f}")
        QMessageBox.information(self,"Single Evaluation Results","\n".join(res))
        # Also refresh previews in case mask changed
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
        self.worker=BatchWorker(self.student_dir, self.expected_dir, self.expected_image_path or "", rules, self.mask_path or None)
        self.worker.progress.connect(self.on_worker_progress); self.worker.finished.connect(lambda res: self.on_batch_finished(res, save))
        self.worker.start(); self.progress.exec_()

    def on_worker_progress(self, percent, fname):
        self.progress.setValue(percent); self.progress.setLabelText(f"Processing: {fname}"); QApplication.processEvents()

    def on_batch_finished(self, results, save_path):
        try: self.progress.setValue(100); self.progress.close()
        except: pass
        if results:
            keys=set(); [keys.update(r.keys()) for r in results]; keys=["student"]+[k for k in sorted(keys) if k!="student"]
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
            restored_mask = ""
            for r in rules:
                notes = r.get("notes","")
                # capture exp_cout if present (kept from your previous version)
                if r.get("expected_cout"):
                    prefix = f"exp_cout={r.get('expected_cout')}"
                    notes = f"{prefix},{notes}" if notes else prefix
                # detect mask=... and restore overlay
                mpath = extract_mask_from_notes(notes)
                if (not restored_mask) and mpath and os.path.exists(mpath):
                    restored_mask = mpath
                row=self.table.rowCount(); self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(r.get("output",""))); self.table.setItem(row,1,QTableWidgetItem(r.get("metric","")))
                self.table.setItem(row,2,QTableWidgetItem(str(r.get("max_score",1.0)))); self.table.setItem(row,3,QTableWidgetItem(r.get("tolerances","")))
                self.table.setItem(row,4,QTableWidgetItem(r.get("mode","bucket"))); self.table.setItem(row,5,QTableWidgetItem(r.get("aggregation","mean")))
                self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem(r.get("result","")))
            if restored_mask:
                self.mask_path = restored_mask
                self.mask_info.setText(f"Mask: {os.path.basename(restored_mask)}")
                self._update_previews()
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
                "expected_cout": nd.get("exp_cout",""),
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
                "expected_cout": nd.get("exp_cout","")
            })
        self.rules_text.setPlainText(json.dumps(rules, indent=2))

    def on_apply_rules_text(self):
        try: rules=json.loads(self.rules_text.toPlainText())
        except Exception as e: QMessageBox.warning(self,"Invalid JSON",f"{e}"); return
        self.table.setRowCount(0)
        for r,rule in enumerate(rules):
            self.table.insertRow(r)
            self.table.setItem(r,0,QTableWidgetItem(rule.get("output",""))); self.table.setItem(r,1,QTableWidgetItem(rule.get("metric","")))
            self.table.setItem(r,2,QTableWidgetItem(str(rule.get("max_score",1.0)))); self.table.setItem(r,3,QTableWidgetItem(rule.get("tolerances","")))
            self.table.setItem(r,4,QTableWidgetItem(rule.get("mode","bucket"))); self.table.setItem(r,5,QTableWidgetItem(rule.get("aggregation","mean")))
            notes = rule.get("notes","")
            if rule.get("expected_cout"):
                prefix=f"exp_cout={rule.get('expected_cout')}"
                notes = f"{prefix},{notes}" if notes else prefix
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
            # include mask reference if currently set
            notes = 'from scan'
            if self.mask_path:
                notes += f",mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem('')); row+=1
        if console:
            self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem('cout')); self.table.setItem(row,1,QTableWidgetItem('exact'))
            self.table.setItem(row,2,QTableWidgetItem('1')); self.table.setItem(row,3,QTableWidgetItem(''))
            self.table.setItem(row,4,QTableWidgetItem('bucket')); self.table.setItem(row,5,QTableWidgetItem(''))  # agg hidden for cout
            notes = 'file=stdout.txt'
            if self.mask_path:
                notes += f",mask={self.mask_path}"
            self.table.setItem(row,6,QTableWidgetItem(notes)); self.table.setItem(row,7,QTableWidgetItem(''))

if __name__ == "__main__":
    app=QApplication(sys.argv); win=ImageGraderApp(); win.show(); sys.exit(app.exec())
