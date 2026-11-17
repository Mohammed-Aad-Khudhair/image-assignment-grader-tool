"""Write EXPECTED_RESULTS.csv and EXPECTED_TOTALS.csv into every task folder from answer_key.json.
Usage: python make_expected_csv.py <course_dir>"""
import sys, os, csv, json, math
C = sys.argv[1]
key = json.load(open(os.path.join(C, "answer_key.json"), encoding="utf-8"))
def fmt(x, nd=6):
    if x is None: return ""
    if isinstance(x, float) and math.isinf(x): return "inf"
    return f"{float(x):.{nd}g}" if abs(float(x)) < 1e6 else f"{float(x):.0f}"
tasks = sorted({k["task"] for k in key})
for t in tasks:
    K = [k for k in key if k["task"] == t]
    with open(os.path.join(C, t, "EXPECTED_RESULTS.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["student", "rule", "metric", "expected_value", "expected_status", "expected_score", "max_score"])
        for k in K:
            w.writerow([k["student"], k["rule"], k["metric"], "" if k["status"] else fmt(k["value"], 6), k["status"] or "",
                        fmt(k["score"], 4), fmt(k["max"])])
    tot = {}
    for k in K:
        s = tot.setdefault(k["student"], [0.0, 0.0]); s[0] += k["score"]; s[1] += float(k["max"])
    with open(os.path.join(C, t, "EXPECTED_TOTALS.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["student", "expected_total", "max_total"])
        for s, (a, b) in tot.items(): w.writerow([s, f"{a:.4g}", fmt(b)])
print("written for", len(tasks), "tasks")
