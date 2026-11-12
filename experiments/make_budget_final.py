"""Builds the final E1 candidate set from validated outputs only (no simulation): results/budget_*_final.csv =
the matched sweep (g in {0.005, ..., 0.10}, all cells) plus the extension run (TAG gainext: cumulative = g 0, 0.0005, 0.001,
0.002; constant and drift; its g = 0.005 duplicates, which reproduce the matched sweep exactly, are dropped). Timing and jump
paths are unchanged and copied from the matched run. budget_mpl.md §25.
Usage (from experiments/): python3 make_budget_final.py"""
import numpy as np
import pandas as pd

FAMS = ["gaussian", "clayton", "gumbel", "t"]
old = pd.read_csv("results/budget_sweep_matched.csv")
new = pd.concat([pd.read_csv(f"results/budget_sweep_gainext_{f}.csv") for f in FAMS])
fin = pd.concat([old, new[~np.isclose(new.gain, 0.005)]], ignore_index=True)
assert not fin.duplicated(["family", "scenario", "split", "estimator", "approx", "gain"]).any()
fin.to_csv("results/budget_sweep_final.csv", index=False)
olds = pd.read_csv("results/budget_seeds_matched.csv")
news = pd.concat([pd.read_csv(f"results/budget_seeds_gainext_{f}.csv") for f in FAMS])
pd.concat([olds, news[~np.isclose(news.gain, 0.005)]], ignore_index=True).to_csv("results/budget_seeds_final.csv", index=False)
for k in ("timing", "paths"):
    pd.read_csv(f"results/budget_{k}_matched.csv").to_csv(f"results/budget_{k}_final.csv", index=False)
print(f"budget_sweep_final.csv: {len(fin)} rows ({len(old)} matched + {len(fin) - len(old)} extension); gains {sorted(fin.gain.unique())}")
