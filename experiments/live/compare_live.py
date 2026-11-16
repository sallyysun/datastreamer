"""Live-pipeline rows (results/live_seeds[_fam].csv, run_live.sh) against the in-process probe's stored rows
(../6.4-copula-density/results/strategies_seeds_mix[_fam].csv, probe_strategies MIXDENS=1 TAG=mix), matched on
(copula, scenario, estimator, strategy, seed, window).
Usage: python3 compare_live.py -> results/live_vs_probe.txt"""
import os
import pandas as pd

K = ["copula", "scenario", "estimator", "strategy", "seed", "window"]
L, P = [], []
for f, s in [("gaussian", ""), ("clayton", "_clayton"), ("gumbel", "_gumbel"), ("t", "_t")]:
    if os.path.exists(f"results/live_seeds{s}.csv"):
        L.append(pd.read_csv(f"results/live_seeds{s}.csv"))
        P.append(pd.read_csv(f"../6.4-copula-density/results/strategies_seeds_mix{s}.csv"))
L, P = pd.concat(L), pd.concat(P)
m = L.merge(P, on=K, suffixes=("_live", "_probe"), how="left")
out = [f"{len(L)} live rows ({L.seed.nunique()} seeds; {', '.join(sorted(L.copula.unique()))}); "
       f"{m.rmse_probe.notna().sum()} matched to stored probe rows."]
for c in ["rmse", "excess_x1000", "h2_x1000"]:
    d = (m[c + "_live"] - m[c + "_probe"]).abs()
    out.append(f"{c:13s} identical {int((d == 0).sum())}/{len(d)}  max |live - probe| {d.max():.3g}")
open("results/live_vs_probe.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
