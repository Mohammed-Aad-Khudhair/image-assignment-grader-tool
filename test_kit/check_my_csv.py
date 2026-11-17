"""Compare a CSV exported by ImgGrader (Batch Evaluate) with EXPECTED_RESULTS.csv.

Usage:  python check_my_csv.py <task folder> <your exported CSV>
Example: python check_my_csv.py T01_pixel_errors "C:/Users/me/Desktop/T01 batch.csv"
"""
import sys, csv, math, os

task, mine = sys.argv[1], sys.argv[2]
exp = [r for r in csv.DictReader(open(os.path.join(task, "EXPECTED_RESULTS.csv"), encoding="utf-8")) if r.get("student")]
got = {r["student"]: r for r in csv.DictReader(open(mine, encoding="utf-8"))}
ok = bad = 0
for e in exp:
    r = got.get(e["student"]); col = f"R{e['rule']}_{e['metric']}"
    if r is None:
        print(f"MISSING student {e['student']} in your CSV"); bad += 1; continue
    val, sc = r.get(col + "_val", ""), r.get(col + "_score", "")
    if e["expected_status"]:
        good = str(val).startswith(e["expected_status"])
    else:
        x, y = float(val) if val not in ("", None) else None, float(e["expected_value"])
        good = x is not None and ((math.isinf(x) and math.isinf(y)) or abs(x - y) <= 1e-3 * max(1, abs(y)))
    good = good and abs(float(sc or 0) - float(e["expected_score"])) <= 2e-3 * max(1, float(e["max_score"]))
    ok += good; bad += not good
    if not good:
        print(f"DIFFERENT  {e['student']:28s} R{e['rule']} {e['metric']:14s} yours: value={val} score={sc}"
              f"   expected: {e['expected_status'] or e['expected_value']} score={e['expected_score']}")
print(f"\n{ok} of {ok + bad} results match the expected results.")
