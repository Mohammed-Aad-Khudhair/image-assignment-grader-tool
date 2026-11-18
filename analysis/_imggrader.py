"""Shared helpers for the analysis scripts: load ImgGrader and run its batch evaluation."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")

_spec = importlib.util.spec_from_file_location("imggrader", os.path.join(ROOT, "GraderApplicationCode.py"))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)
app = g.QApplication.instance() or g.QApplication(sys.argv)

# The four case-study gradings: (name, case folder, rule file)
CASES = [
    ("ISIC", os.path.join(ROOT, "case_studies", "isic"), "rules.json"),
    ("H01", os.path.join(ROOT, "case_studies", "hrf"), "rules_H01.json"),
    ("G12", os.path.join(ROOT, "case_studies", "hrf"), "rules_G12.json"),
    ("DR07", os.path.join(ROOT, "case_studies", "hrf"), "rules_DR07.json"),
]


def load_rules(path):
    """Same as Import Rules (JSON) in the program."""
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


def run_batch(student_dir, expected_dir, rules, rules_base_dir):
    """Batch Evaluate (student dir) with an Expected Dir, as in the program.
    Returns the list of result rows (one dict per student)."""
    worker = g.BatchWorker(student_dir, expected_dir, "", rules, None, rules_base_dir)
    out = {}
    worker.finished.connect(lambda r: out.setdefault("r", r))
    worker.run()
    return out["r"]


def rule_keys(rules):
    """Column prefix of every rule in the result rows (R<n>_<metric>)."""
    return [f"R{i}_{r['metric']}" for i, r in enumerate(rules, 1)]


def write_csv(name, rows):
    import csv
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
