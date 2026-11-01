"""The eta = 1/32 squared bank (bank_sq32): regret against 32 B_{T,m}(delta) from probe_bank_regret, and
rho_S RMSE (x1e-3) against the eta = 1 squared bank, the MoM log bank, uniform and the best fixed memory
from probe_thesis_bank. Paired mean +- 2 SE over seeds. Note: notes/studies/thesis_bank_sq32.md.
Usage (from experiments/): python3 sq32_table.py"""
import pandas as pd

fams = ["", "_clayton", "_gumbel", "_t"]
r = pd.concat(pd.read_csv(f"results/bank_regret_42{s}.csv") for s in fams)
print("## Regret, squared banks (bound = B/eta)\n")
print("| bank | delta | scenario | m | mean | 95% | max | bound | max/bound | viol |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for b, dl in [("sq", "1e-4"), ("sq32", "1e-4"), ("sq32_dstar", "2.5e-4")]:
    for (sc, m), g in r[r.bank == b].groupby(["scenario", "m"]):
        if m not in (0, 1, 2, 5, 20): continue
        print(f"| {b} | {dl} | {sc} | {m} | {g.regret.mean():.2f} | {g.regret.quantile(.95):.2f} | {g.regret.max():.2f} | "
              f"{g.bound.mean():.1f} | {(g.regret / g.bound).max():.2f} | {(g.regret > g.bound).sum()}/{len(g)} |")

d = pd.concat(pd.read_csv(f"results/thesis_bank{s}.csv") for s in fams)

def ci(x):
    x = x.dropna()
    return f"{x.mean():+.1f} ± {2 * x.std(ddof=1) / len(x) ** .5:.1f}"

for win in ["whole", "rec_0_100", "rec_100_500"]:
    w = d[d.window == win].pivot_table(index=["copula", "scenario", "component", "seed"], columns="method", values="rmse") * 1000
    print(f"\n## RMSE x1e-3, window {win}: bank_sq32 minus ...\n")
    print("| copula | scenario | component | sq32 | − bank (η=1) | − bank_log | − uniform | − best fixed |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (c, s, k), g in w.groupby(level=[0, 1, 2]):
        best = g[[x for x in g.columns if x.startswith("fixed_")]].mean().idxmin()
        print(f"| {c} | {s} | {k} | {g.bank_sq32.mean():.1f} | {ci(g.bank_sq32 - g.bank)} | {ci(g.bank_sq32 - g.bank_log)} | "
              f"{ci(g.bank_sq32 - g.uniform)} | {ci(g.bank_sq32 - g[best])} ({best[6:]}) |")
    if win == "whole":
        p = w[w.index.get_level_values(1).isin(["drift", "jump"])]
        print("\nW/L/T per (copula, scenario, component) cell, drift+jump, sq32 vs X (W = sq32 lower, 2 SE):")
        for ref in ["bank", "bank_log", "uniform"]:
            wl = [0, 0, 0]
            for _, g in p.groupby(level=[0, 1, 2]):
                x = g.bank_sq32 - g[ref]
                h = 2 * x.std(ddof=1) / len(x) ** .5
                wl[0 if x.mean() < -h else 1 if x.mean() > h else 2] += 1
            print(f"  vs {ref}: {wl[0]}/{wl[1]}/{wl[2]}")
