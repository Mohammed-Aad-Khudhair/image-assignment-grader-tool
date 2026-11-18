"""How much do MSE, MAE and ΔE depend on the aggregation method?

MSE, MAE and ΔE are first computed per pixel and then reduced to one number with
the rule's aggregation: mean, median, max or trim10 (mean after removing the
lowest and highest 10 % of the pixel values). This script computes all four for
the grey-level and colour submissions of the test kit (T02 perceptual
distortions, T03 colour errors) with ImgGrader's own metric code, and ranks the
students under each aggregation.

Output: results/aggregation.csv
"""
import os

from scipy.stats import kendalltau

from _imggrader import ROOT, g, write_csv

TASKS = [("T02_perceptual", "coffee.png", "coffee_stud.png"),
         ("T03_colour", None, None)]
AGGS = ("mean", "median", "max", "trim10")


def items():
    for task, ref_name, stud_name in TASKS:
        tdir = os.path.join(ROOT, "test_kit", task)
        ref_name = ref_name or sorted(os.listdir(os.path.join(tdir, "expected")))[0]
        ref = g.imread(os.path.join(tdir, "expected", ref_name))
        sname = stud_name or ref_name.replace(".png", "_stud.png")
        for s in sorted(os.listdir(os.path.join(tdir, "students"))):
            p = os.path.join(tdir, "students", s, sname)
            img = g.imread(p) if os.path.exists(p) else None
            if img is not None and img.shape == ref.shape:
                yield task, s, ref, img


def main():
    rows = []
    data = list(items())
    for metric in ("MSE", "MAE", "ΔE"):
        for task in sorted({t for t, *_ in data}):
            sub = [(s, ref, img) for t, s, ref, img in data if t == task]
            vals = {agg: [g.compute_metric(metric, ref, img, None, agg)[0] for _, ref, img in sub] for agg in AGGS}
            for agg in AGGS:
                tau = kendalltau(vals["mean"], vals[agg]).statistic
                for (s, _, _), v in zip(sub, vals[agg]):
                    rows.append({"task": task, "metric": metric, "aggregation": agg, "student": s,
                                 "value": round(float(v), 6), "kendall_tau_b_vs_mean": round(float(tau), 6)})
    print("wrote", write_csv("aggregation.csv", rows))


if __name__ == "__main__":
    main()
