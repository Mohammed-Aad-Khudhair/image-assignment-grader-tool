# make_students.py
# Generate student vessel masks (S1..S4) for all HRF images you placed under expected/.
# Expected files per image ID (e.g., H01, DR07, G12):
#   expected/ID.jpg                # input RGB (for realism only; not used here)
#   expected/ID_vessels_gt.png     # binary GT (0/255)
#   expected/ID_fov.png            # FOV mask (0/255); if missing, uses full image
#
# Outputs per ID:
#   students/s1/ID_vessels_stud.png (+ stdout.txt with time_ms, algo)
#   students/s2/...
#   students/s3/...
#   students/s4/...
#
# S1: near-correct
# S2: under-seg (miss thin vessels)   -> erosion/opening
# S3: over-seg (false positives)      -> dilation
# S4: clearly incorrect                -> shift + erosion
#
# IoU tuning uses the FOV mask so scores reflect what the grader will compute.

from pathlib import Path
import re, time, random
import numpy as np
import cv2

BASE = Path(__file__).resolve().parent
EXP  = BASE / "expected"
STUD = BASE / "students"
for sub in ["s1","s2","s3","s4"]:
    (STUD/sub).mkdir(parents=True, exist_ok=True)

# ---- helper io ----
def read_bin(p: Path, fallback_shape=None):
    if not p.exists():
        if fallback_shape is None:
            raise FileNotFoundError(p)
        return np.ones(fallback_shape, np.uint8)*255
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    return im

def write_bin(p: Path, m: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    m = (m>0).astype(np.uint8)*255
    cv2.imwrite(str(p), m)

def write_log(dirpath: Path, algo: str):
    dirpath.mkdir(parents=True, exist_ok=True)
    ms = random.randint(500, 1800)
    (dirpath/"stdout.txt").write_text(f"time_ms={ms}\nalgo={algo}\n", encoding="utf-8")

# ---- metrics (in-FOV) ----
def iou_in_mask(gt: np.ndarray, pred: np.ndarray, fov: np.ndarray) -> float:
    A = (gt>0) & (fov>0)
    B = (pred>0) & (fov>0)
    inter = np.logical_and(A,B).sum()
    union = np.logical_or (A,B).sum()
    return inter / max(1, union)

# ---- morph recipes ----
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def tune_to_target(gt, fov, mode="erode", target=0.97, iters_max=5):
    """Morph gt until IoU_in_FOV ~ target (move in the right direction)."""
    cand = gt.copy()
    last = iou_in_mask(gt, cand, fov)
    for it in range(1, iters_max+1):
        if mode == "erode":
            cand = cv2.erode(cand, K3, iterations=1)
        elif mode == "dilate":
            cand = cv2.dilate(cand, K3, iterations=1)
        elif mode == "open":
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3, iterations=1)
        else:
            break
        cur = iou_in_mask(gt, cand, fov)
        # stop when we've crossed or reached the target (going down from 1.0)
        if cur <= target + 1e-4:
            return cand, cur, it
        last = cur
    return cand, last, iters_max

def translate(mask, dx, dy):
    h, w = mask.shape[:2]
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)

# ---- discover IDs ----
ids = []
for gt_path in sorted(EXP.glob("*_vessels_gt.png")):
    m = re.match(r"(.+)_vessels_gt\.png$", gt_path.name)
    if m:
        ids.append(m.group(1))

if not ids:
    raise SystemExit("No *_vessels_gt.png found in expected/")

print(f"Found {len(ids)} image(s):", ", ".join(ids))

report = []
t0_all = time.time()

for ID in ids:
    gt_path  = EXP / f"{ID}_vessels_gt.png"
    fov_path = EXP / f"{ID}_fov.png"
    img_path = None
    # try common image extensions for the input RGB (optional)
    for ext in [".jpg",".jpeg",".png",".tif",".tiff"]:
        p = EXP / f"{ID}{ext}"
        if p.exists():
            img_path = p; break

    gt  = read_bin(gt_path)
    fov = read_bin(fov_path, fallback_shape=gt.shape)

    # --- S1: near-correct (light smoothing & close) ---
    s1 = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, K3, iterations=1)
    s1 = cv2.GaussianBlur(s1, (3,3), 0.6)
    _, s1 = cv2.threshold(s1, 127, 255, cv2.THRESH_BINARY)
    iou1 = iou_in_mask(gt, s1, fov)

    # --- S2: under-seg (drop thin vessels via opening/erosion) ---
    s2, iou2, it2 = tune_to_target(gt, fov, mode="open",   target=0.955, iters_max=4)
    if iou2 > 0.96:
        s2, iou2, it2b = tune_to_target(s2, fov, mode="erode", target=0.955, iters_max=2)

    # --- S3: over-seg (thicken vessels; slight dilation) ---
    s3, iou3, it3 = tune_to_target(gt, fov, mode="dilate", target=0.935, iters_max=4)

    # --- S4: clearly wrong (shift + erosion) ---
    s4 = translate(gt, dx=10, dy=-7)
    s4 = cv2.erode(s4, K5, iterations=2)
    iou4 = iou_in_mask(gt, s4, fov)

    # save masks & logs
    for sid, mask, algo in [
        ("s1", s1, "cleanup"),
        ("s2", s2, "opening"),
        ("s3", s3, "dilate"),
        ("s4", s4, "shifted"),
    ]:
        out = STUD/sid/f"{ID}_vessels_stud.png"
        write_bin(out, mask)
        write_log(STUD/sid, algo)

    report.append((ID, iou1, iou2, iou3, iou4))
    print(f"{ID}: IoU(FOV) -> S1 {iou1:.4f}, S2 {iou2:.4f}, S3 {iou3:.4f}, S4 {iou4:.4f}")

dt = (time.time()-t0_all)*1000
print(f"\nWrote students for {len(ids)} image(s). Elapsed ~{dt:.0f} ms total.")

print("\nSummary (IoU in FOV):")
for ID, a,b,c,d in report:
    print(f"  {ID}: S1={a:.4f}  S2={b:.4f}  S3={c:.4f}  S4={d:.4f}")
