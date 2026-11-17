"""Run the real ImgGrader GUI on the independent test course and compare with the answer key.

Only modal dialogs are stubbed (they return the chosen file/folder); every
button handler of the real application runs.
Usage: python run_suite.py <GraderApplicationCode.py> <course_dir> <out_dir>
"""
import os, sys, csv, json, math, shutil, importlib.util
os.environ["QT_QPA_PLATFORM"] = "offscreen"

GR, C, OUT = (os.path.abspath(x) for x in sys.argv[1:4])
spec = importlib.util.spec_from_file_location("grader", GR)
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
app = g.QApplication.instance() or g.QApplication(sys.argv)
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
key = json.load(open(os.path.join(C, "answer_key.json"), encoding="utf-8"))
design = json.load(open(os.path.join(C, "design.json"), encoding="utf-8"))

NEXT = {"open": [], "save": [], "dir": []}; MSG = []
g.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (NEXT["open"].pop(0), ""))
g.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (NEXT["save"].pop(0), ""))
g.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: NEXT["dir"].pop(0))
g.QMessageBox.information = staticmethod(lambda *a, **k: MSG.append(("info",) + tuple(a[1:3])))
g.QMessageBox.warning = staticmethod(lambda *a, **k: MSG.append(("warn",) + tuple(a[1:3])))

results = []          # (section, task, student, rule, ok, detail)
def rec(section, task, student, rule, ok, detail=""):
    results.append((section, task, student, rule, bool(ok), detail))

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def same_value(got, k):
    if k["status"] is not None:
        return isinstance(got, str) and got.startswith(k["status"]), f"expected status '{k['status']}', got {got!r}"
    v = num(got)
    if v is None: return False, f"expected value {k['value']!r}, got {got!r}"
    if math.isinf(k["value"]) or math.isinf(v): return v == k["value"], f"{v} vs {k['value']}"
    return abs(v - k["value"]) <= k["value_tol"] * max(1.0, abs(k["value"])), f"got {v!r}, expected {k['value']!r}"

def table(win):
    """Rule table; MaxScore compared as a number ('4' and '4.0' are the same score)."""
    T = win.table
    rows = [[(T.item(r, c).text() if T.item(r, c) else "") for c in range(7)] for r in range(T.rowCount())]
    for row in rows:
        try: row[2] = float(row[2])
        except ValueError: pass
    return rows

os.chdir("/")                                            # the app is started from an unrelated folder
for tname, t in design.items():
    T = os.path.join(C, tname); K = [k for k in key if k["task"] == tname]
    win = g.ImageGraderApp()
    # --- Import Rules (JSON) button
    NEXT["open"].append(os.path.join(T, "rules.json")); win.on_import_json()
    src_rules = json.load(open(os.path.join(T, "rules.json"), encoding="utf-8"))
    imported = table(win)
    rec("rule import", tname, "-", "-", len(imported) == len(src_rules) and all(
        row[0] == r["output"] and row[1] == r["metric"] and float(row[2]) == float(r["max_score"]) and row[3] == r["tolerances"]
        and row[6] == (r["notes"] or "") for row, r in zip(imported, src_rules)), f"{len(imported)} rows")
    # --- Export -> Import round trip, Refresh Text -> Apply Text round trip
    exp_json = os.path.join(OUT, f"{tname}_exported.json"); NEXT["save"].append(exp_json); win.on_export_rules()
    NEXT["open"].append(exp_json); win.on_import_json()
    rec("export/import round trip", tname, "-", "-", table(win) == imported)
    win.on_refresh_rules_text(); txt = win.rules_text.toPlainText(); win.table.setRowCount(0)
    win.rules_text.setPlainText(txt); win.on_apply_rules_text()
    rec("text round trip", tname, "-", "-", table(win) == imported)
    NEXT["open"].append(os.path.join(T, "rules.json")); win.on_import_json()      # base folder for relative masks
    # --- Select Expected Dir / Student Dir, Batch Evaluate
    NEXT["dir"] += [os.path.join(T, "expected"), os.path.join(T, "students")]
    win.on_pick_expected_dir(); win.on_pick_student_dir()
    picker = sorted(win.student_picker.itemText(i) for i in range(win.student_picker.count()))
    rec("student list", tname, "-", "-", picker == sorted(t["students"]), f"{picker}")
    csv_path = os.path.join(OUT, f"{tname}_batch.csv"); NEXT["save"].append(csv_path)
    win.on_batch_eval()
    rows = {r["student"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    for k in K:
        r = rows.get(k["student"])
        if r is None:
            rec("batch value", tname, k["student"], k["rule"], False, "student missing in CSV"); continue
        col = f"R{k['rule']}_{k['metric']}"
        ok, det = same_value(r.get(col + "_val"), k)
        rec("batch value/status", tname, k["student"], k["rule"], ok, f"{k['metric']}: {det}")
        sc = num(r.get(col + "_score"))
        rule_def = t["rules"][k["rule"] - 1]
        # continuous scores are proportional to the value, so they inherit the value tolerance
        s_tol = 1e-9 if rule_def["mode"] != "continuous" or k["value"] is None else max(1e-9, 2 * k["value_tol"] * float(k["max"]))
        rec("batch score", tname, k["student"], k["rule"], sc is not None and abs(sc - k["score"]) <= s_tol,
            f"{k['metric']}: got {sc}, expected {k['score']}")
        if k["resized"] is not None:
            rec("batch resized flag", tname, k["student"], k["rule"], str(r.get(col + "_resized")) == str(k["resized"]),
                f"got {r.get(col + '_resized')}, expected {k['resized']}")
    for s in t["students"]:
        exp_total = sum(k["score"] for k in K if k["student"] == s)
        t_tol = sum(max(1e-9, 2 * k["value_tol"] * float(k["max"])) for k in K if k["student"] == s)
        rec("batch total", tname, s, "-", s in rows and abs(float(rows[s]["total_score"]) - exp_total) <= t_tol,
            f"got {rows.get(s, {}).get('total_score')}, expected {exp_total}")
    # --- explicit expected image mode must give the same result (image tasks)
    if t["expected_files"]:
        win.expected_image_path = os.path.join(T, "expected", t["expected_files"][0])
        csv2 = os.path.join(OUT, f"{tname}_batch_explicit.csv"); NEXT["save"].append(csv2); win.on_batch_eval()
        rows2 = {r["student"]: r for r in csv.DictReader(open(csv2, encoding="utf-8"))}
        same = all(rows2[s][c] == rows[s][c] for s in rows for c in rows[s] if c.endswith(("_val", "_score")) or c == "total_score")
        rec("explicit image = expected dir", tname, "-", "-", same)
    # --- Run Single Eval for every student
    for s, sp in t["students"].items():
        imgs = [f for f in sp["filenames"] if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if t["expected_files"]:
            if sp.get("status") or len(imgs) != 1: continue            # Single Eval needs one chosen image
            win.expected_image_path = os.path.join(T, "expected", t["expected_files"][0])
            win.student_image_path = os.path.join(T, "students", s, imgs[0])
        else:
            win.student_picker.setCurrentText(s)
        win.on_run_single()
        cells = [win.table.item(i, 7).text() if win.table.item(i, 7) else "" for i in range(win.table.rowCount())]
        for k in K:
            if k["student"] != s: continue
            cell = cells[k["rule"] - 1]
            if k["status"] is not None:
                ok = k["status"].split(":")[0] in cell or k["status"] in cell
                det = f"expected '{k['status']}', got {cell!r}"
            elif tname == "T08_console":
                want = f"cout: {k['value']:.1f} | score {k['score']:.2f}"; ok = cell == want; det = f"{cell!r} vs {want!r}"
            else:
                v = k["value"]; vs = "inf" if math.isinf(v) else f"{v:.4f}"
                want = f"Value {vs} | Score {k['score']:.2f}"
                ok = cell == want
                if not ok and cell.startswith("Value ") and k["metric"] in ("ΔE", "SSIM"):   # 4th decimal may round differently
                    got_v = num(cell.split()[1]); got_s = cell.split("Score ")[-1]
                    ok = got_v is not None and abs(got_v - v) <= max(k["value_tol"], 1e-4) * max(1, abs(v)) and got_s == f"{k['score']:.2f}"
                det = f"{cell!r} vs {want!r}"
            rec("single eval", tname, s, k["rule"], ok, f"{k['metric']}: {det}")
    print(f"{tname}: done", flush=True)

# ------------------------------------------------------------------ report
by = {}
for sec, *_ , ok, _d in results:
    by.setdefault(sec, [0, 0]); by[sec][0] += 1; by[sec][1] += ok
lines = [f"{'check':32s} {'total':>6s} {'passed':>7s}"] + [f"{s:32s} {n:6d} {p:7d}" for s, (n, p) in by.items()]
tot = sum(n for n, _ in by.values()); passed = sum(p for _, p in by.values())
lines.append(f"{'ALL':32s} {tot:6d} {passed:7d}")
fails = [r for r in results if not r[4]]
lines.append(f"\nfailures: {len(fails)}")
lines += [f"FAIL [{s}] {t} / {st} / R{r}: {d}" for s, t, st, r, ok, d in fails[:200]]
open(os.path.join(OUT, "suite_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
with open(os.path.join(OUT, "suite_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["check", "task", "student", "rule", "passed", "detail"]); w.writerows(results)
print("\n".join(lines[:len(by) + 3])); print("\n".join(lines[len(by) + 3:len(by) + 60]))
