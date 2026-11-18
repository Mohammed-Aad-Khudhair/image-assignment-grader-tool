"""How much do the case-study totals depend on the chosen tolerances and scoring mode?

1. Every tolerance of every rule is made stricter or more lenient by 10 % and 20 %
   and the four case studies are graded again with ImgGrader.
   - Lower-is-better metrics (N_DIFF_PIXELS, Objects, ...): t -> t * (1 + d).
   - Higher-is-better metrics bounded by 1 (IoU, SSIM, NCC, Contours,
     ContoursStrict): the allowed error 1 - t is scaled, t -> 1 - (1 - t) * (1 + d).
   d > 0 is more lenient, d < 0 is stricter. The per-pixel threshold of
   N_DIFF_PIXELS (thr:10) is not changed.
2. All rules are switched from bucket to continuous scoring.

For each variant the totals of the four students and Kendall's tau-b between the
original and the new ranking are reported.

Output: results/thresholds.csv
"""
import re

from scipy.stats import kendalltau

from _imggrader import CASES, load_rules, run_batch, write_csv
import os

BOUNDED_HIGHER = {"IoU", "SSIM", "NCC", "Contours", "ContoursStrict"}


def perturb(rule, d):
    tol = rule["tolerances"]
    prefix = ""
    if rule["metric"] == "N_DIFF_PIXELS":
        m = re.match(r"\s*(thr\s*[:=]\s*\d+\s*\|?)\s*(.*)$", tol, re.I)
        if m:
            prefix, tol = m.group(1) + " ", m.group(2)
    vals = [float(t) for t in tol.split(",") if t.strip()]
    if rule["metric"] in BOUNDED_HIGHER:
        new = [1 - (1 - t) * (1 + d) for t in vals]
    else:
        new = [t * (1 + d) for t in vals]
    return {**rule, "tolerances": prefix + ", ".join(f"{t:.6g}" for t in new)}


def totals(case_dir, rules, rule_file):
    rows = run_batch(os.path.join(case_dir, "students"), os.path.join(case_dir, "expected"), rules, case_dir)
    return {r["student"]: float(r["total_score"]) for r in rows}


def main():
    out = []
    for name, case_dir, rule_file in CASES:
        rules = load_rules(os.path.join(case_dir, rule_file))
        base = totals(case_dir, rules, rule_file)
        students = list(base)
        variants = [(f"tolerances {d:+.0%}", [perturb(r, d) for r in rules]) for d in (-0.2, -0.1, 0.1, 0.2)]
        variants.append(("continuous mode", [{**r, "mode": "continuous"} for r in rules]))
        for label, new_rules in variants:
            new = totals(case_dir, new_rules, rule_file)
            tau = kendalltau([base[s] for s in students], [new[s] for s in students]).statistic
            for s in students:
                out.append({"case": name, "variant": label, "student": s,
                            "total_original": round(base[s], 6), "total_variant": round(new[s], 6),
                            "change": round(new[s] - base[s], 6), "kendall_tau_b": round(float(tau), 6)})
        print(name, "done")
    print("wrote", write_csv("thresholds.csv", out))


if __name__ == "__main__":
    main()
