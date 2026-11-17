"""Answer key from the oracle (documentation-based, no grader code).
Also checks the oracle against every value and status known by design.
Usage: python make_key.py <course_dir>
"""
import os, sys, json, math
import oracle as O

C = sys.argv[1]
design = json.load(open(os.path.join(C, "design.json"), encoding="utf-8"))
key, design_checks, design_fail = [], 0, []

def close(a, b, tol):
    if isinstance(a, float) and isinstance(b, float) and math.isinf(a) and math.isinf(b): return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))

for tname, t in design.items():
    T = os.path.join(C, tname)
    exp_path = os.path.join(T, "expected", t["expected_files"][0]) if t["expected_files"] else None
    exp = O.read(exp_path) if exp_path else None
    for sname, sp in t["students"].items():
        sdir = os.path.join(T, "students", sname)
        imgs = [f for f in sp["filenames"] if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        last_file = None
        for ri, r in enumerate(t["rules"], 1):
            row = dict(task=tname, student=sname, rule=ri, metric=r["metric"], max=r["max_score"],
                       value=None, status=None, score=0.0, value_tol=1e-9, resized=None, boundary=False)
            try:
                if r["output"].lower() == "cout":
                    row["value"] = O.console(sdir, r["metric"], r["tolerances"], r["notes"])
                    row["score"] = float(r["max_score"]) if row["value"] >= 1 else 0.0
                    raise StopIteration
                if sp.get("status"):
                    raise O.Expected(sp["status"])
                if r["output"]:
                    if len(imgs) != 1: raise O.Expected("MISSING student")
                    last_file = os.path.join(sdir, imgs[0])
                stud = O.read(last_file)
                if stud is None: raise O.Expected("UNREADABLE student")
                stud_al, row["resized"] = O.align(exp, stud, r["metric"])
                mask = O.re.search(r"mask\s*=\s*([^,;]+)", r["notes"] or "")
                roi = None
                if mask and r["metric"] != "SIZE":
                    roi = O.roi_from_file(os.path.normpath(os.path.join(T, mask.group(1).strip())), exp.shape[:2])
                if r["metric"] == "SIZE": stud_al = stud
                v = O.metric(r["metric"], exp, stud_al, roi, r.get("aggregation") or "mean", r["tolerances"], r["notes"])
                if r["metric"] == "N_DIFF_PIXELS": tol = O.ndiff_params(r["tolerances"], r["notes"])[3]
                else: tol = r["tolerances"]
                row["value"] = v
                row["value_tol"] = {"SSIM": 1e-7, "ΔE": 2e-3, "NCC": 1e-9}.get(r["metric"], 1e-9)
                row["boundary"] = O.near_boundary(r["metric"], v, tol, 1e-6 if r["metric"] not in ("ΔE",) else 5e-3)
                row["score"] = O.score(r["metric"], v, float(r["max_score"]), tol, r["mode"])
            except StopIteration:
                pass
            except O.Expected as e:
                row["status"], row["score"], row["value"] = e.status, 0.0, None
            # ---- check the oracle against the design ----
            d = sp.get("designed", {}).get(str(ri))
            if d is not None:
                design_checks += 1
                if row["value"] is None or not close(row["value"], float(d), max(row["value_tol"], 1e-9)):
                    design_fail.append((tname, sname, ri, r["metric"], d, row["value"], row["status"]))
            ds = sp.get("status_by_rule", {}).get(str(ri)) or sp.get("status")
            if ds is not None:
                design_checks += 1
                if not (row["status"] or "").startswith(ds):
                    design_fail.append((tname, sname, ri, r["metric"], ds, row["value"], row["status"]))
            key.append(row)

json.dump(key, open(os.path.join(C, "answer_key.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(f"answer key: {len(key)} student-rule results; boundary cases: {sum(r['boundary'] for r in key)}")
print(f"oracle vs design: {design_checks} checks, {len(design_fail)} failed")
for f in design_fail: print("  DESIGN MISMATCH", f)
