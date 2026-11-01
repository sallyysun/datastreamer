"""Pick eta, delta for ForgettingBank from probe_bank_tune output.
Objective: mean over (family, scenario in drift/jump, component, seed) of whole-run RMSE minus the
cell's hindsight-best fixed memory, x1e-3.
Usage (from experiments/): python3 tune_bank_table.py <first_seed> [metric]"""
import sys
import pandas as pd

first = sys.argv[1] if len(sys.argv) > 1 else "1"
metric = sys.argv[2] if len(sys.argv) > 2 else "rmse"
sc = 1000 if metric == "rmse" else 1
d = pd.concat(pd.read_csv(f"results/bank_tune_{first}{s}.csv") for s in ["", "_clayton", "_gumbel", "_t"])
d = d[d.window == "whole"]
cell = ["copula", "scenario", "component"]
fx = d[d.method.str.startswith("fixed")]
best = fx.groupby(cell + ["method"])[metric].mean().groupby(level=[0, 1, 2]).min().rename("best")
b = d[d.method.isin(["sq", "log", "own"])].groupby(cell + ["method", "eta", "delta"])[metric].mean().reset_index()
b = b.merge(best.reset_index(), on=cell)
b["reg"] = (b[metric] - b.best) * sc
pd.set_option("display.width", 250)
for m in ["sq", "log", "own"]:
    g = b[b.method == m]
    print(f"\n== {m}: mean regret over all 24 cells (rows eta, cols delta)")
    print(g.pivot_table(index="eta", columns="delta", values="reg", aggfunc="mean").round(2).to_string())
    for by in ["component", "scenario"]:
        p = g.groupby([by, "eta", "delta"]).reg.mean().reset_index()
        top = p.loc[p.groupby(by).reg.idxmin()]
        print(f"best per {by}:\n" + top.assign(reg=top.reg.round(2)).to_string(index=False))
