"""Grade the paper's two case studies with ImgGrader and write the CSV files.

Usage (from the repository root):

    python case_studies/reproduce.py

For every rule file it does what Batch Evaluate does in the program
(Import Rules, Expected Dir, Student Dir), using ImgGrader's own batch code,
and writes one CSV per rule file to case_studies/results/ in the same format
as the program's CSV export. File paths in the CSV are relative to the case
folder, so the files are the same on every computer.
"""
import csv
import importlib.util
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")

spec = importlib.util.spec_from_file_location("imggrader", os.path.join(ROOT, "GraderApplicationCode.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
app = g.QApplication.instance() or g.QApplication(sys.argv)

# (case folder, rule file, output CSV)
CASES = [
    ("isic", "rules.json", "isic.csv"),
    ("hrf", "rules_H01.json", "hrf_H01.csv"),
    ("hrf", "rules_G12.json", "hrf_G12.csv"),
    ("hrf", "rules_DR07.json", "hrf_DR07.csv"),
]


def load_rules(path):
    # Same as Import Rules (JSON) in the program.
    import json
    with open(path, encoding="utf-8") as f:
        rules = json.load(f)
    return [{
        "output": r.get("output", ""),
        "metric": r.get("metric", ""),
        "max_score": float(r.get("max_score", 1.0)),
        "tolerances": r.get("tolerances", ""),
        "mode": r.get("mode", "bucket"),
        "aggregation": r.get("aggregation", "mean") or "mean",
        "notes": r.get("notes", "") or "",
    } for r in rules]


os.makedirs(OUT, exist_ok=True)
for case, rule_file, out_name in CASES:
    os.chdir(os.path.join(HERE, case))
    rules = load_rules(rule_file)
    worker = g.BatchWorker("students", "expected", "", rules, None, "")
    res = {}
    worker.finished.connect(lambda r: res.setdefault("r", r))
    worker.run()
    results = res["r"]
    # Mask paths are resolved to absolute paths; write all paths relative to the
    # case folder and with "/", so the files are the same on Windows and Linux.
    case_dir = os.getcwd()
    for r in results:
        for k, v in r.items():
            if k.endswith(("_mask", "_path")) and isinstance(v, str) and v:
                if os.path.isabs(v) and v.startswith(case_dir):
                    v = os.path.relpath(v, case_dir)
                r[k] = v.replace(os.sep, "/")
    # Column order as in the program's CSV export (on_batch_finished).
    keys = list(dict.fromkeys(k for r in results for k in r.keys()))
    keys = ["student"] + [k for k in keys if k not in ("student", "total_score")] + ["total_score"]
    out_path = os.path.join(OUT, out_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    totals = ", ".join(f"{r['student']} {float(r['total_score']):g}" for r in results)
    print(f"{out_name}: {totals}")
