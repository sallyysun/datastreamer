"""Paired tables for probe_thesis_bank: each method minus a reference, x1e-3, mean +- 2 SE over seeds (paired).
best fixed and best VFF rate are picked per cell in hindsight, by mean.
Usage (from experiments/): python3 thesis_bank_table.py [rmse|excess_x1000] [window]"""
import sys
import pandas as pd

metric = sys.argv[1] if len(sys.argv) > 1 else "rmse"
window = sys.argv[2] if len(sys.argv) > 2 else "whole"
scale = 1000 if metric == "rmse" else 1
frames = [pd.read_csv(f"results/thesis_bank{s}.csv") for s in ["", "_clayton", "_gumbel", "_t"]]
d = pd.concat(frames)
d = d[d.window == window]
w = d.pivot_table(index=["copula", "scenario", "component", "seed"], columns="method", values=metric)

def ci(x):
    x = x.dropna()
    return x.mean(), 2.0 * x.std(ddof=1) / len(x) ** .5

rows = []
for key, g in w.groupby(level=[0, 1, 2]):
    fixed = [c for c in g.columns if c.startswith("fixed_")]
    best = g[fixed].mean().idxmin()
    bv = "vff_slow" if g["vff_slow"].mean() < g["vff_fast"].mean() else "vff_fast"
    r = dict(zip(["copula", "scenario", "component"], key), best_fixed=best,
             best_fixed_val=g[best].mean() * scale)
    for m in ["bank", "bank_log", "vff_slow", "vff_fast"]:
        mu, h = ci((g[m] - g[best]) * scale)
        r[f"{m}-best"] = f"{mu:+.1f}±{h:.1f}"
    for m in ["bank", "bank_log"]:
        mu, h = ci((g[m] - g[bv]) * scale)
        r[f"{m}-vff"] = f"{mu:+.1f}±{h:.1f}"
    r["best_vff"] = bv
    rows.append(r)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(out.to_string(index=False))
