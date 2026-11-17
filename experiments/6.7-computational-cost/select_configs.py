"""Section 6.7: validation-only configuration selection within each storage budget (n = 15,000).
For every copula family x scenario x budget (16, 64, 256 KiB) x estimator, the approximation size and gain with the lowest
validation RMSE (seeds 1000-1039) among candidates whose peak retained state fits the budget; then its test RMSE (seeds 42-141).
Writes results/budget_selected.csv (one row per selection) and results/budget_gain.csv (per family x scenario x estimator, the
percentage RMSE reduction at 64 and 256 KiB relative to the 16 KiB selection, 100 (mean RMSE_16 - mean RMSE_b) / mean RMSE_16 over
the 100 paired test seeds, with its Monte Carlo SE).
Usage: python3 select_configs.py"""
import math
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../common")
import config as cfg  # noqa: E402

BUD = {b: b * 1024 for b in cfg.BUDGETS}
E = ["mpl_dskll", "xiao", "hermite"]
FAMS, SC = cfg.FAMILIES_T31, cfg.SCENARIOS
sw, sd = pd.read_csv("results/budget_sweep_n15k.csv"), pd.read_csv("results/budget_seeds_n15k.csv")
val, tst = sw[sw.split == "val"], sw[sw.split == "test"]
SD = sd.set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()

rows = []
for f in FAMS:
    for s in SC:
        for b, cap in BUD.items():
            for e in E:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
                r = v.loc[v.rmse.idxmin()]
                a, g = int(r.approx), float(r.gain)
                t = tst[(tst.family == f) & (tst.scenario == s) & (tst.estimator == e) & (tst.approx == a) & np.isclose(tst.gain, g)].iloc[0]
                x = SD.loc[(f, s, e, a, g)]
                rows.append(dict(family=f, scenario=s, budget_kib=b, estimator=e, approx=a, gain=g, val_rmse=100 * r.rmse,
                                 rmse=100 * x.mean(), se=100 * x.std(ddof=1) / math.sqrt(len(x)), peak_val=int(r.peak_bytes),
                                 peak_test=int(t.peak_bytes), cap=cap, util=100 * max(r.peak_bytes, t.peak_bytes) / cap))
C = pd.DataFrame(rows)
C.to_csv("results/budget_selected.csv", index=False)

gain = []
for (f, s, e), x in C.groupby(["family", "scenario", "estimator"]):
    x = x.set_index("budget_kib")
    base = SD.loc[(f, s, e, int(x.loc[16].approx), float(x.loc[16].gain))].values
    for b in (64, 256):
        d = base - SD.loc[(f, s, e, int(x.loc[b].approx), float(x.loc[b].gain))].values
        gain.append(dict(family=f, scenario=s, estimator=e, budget=b, pct=100 * d.mean() / base.mean(),
                         pct_mc_se=100 * d.std(ddof=1) / np.sqrt(len(d)) / base.mean()))
pd.DataFrame(gain).to_csv("results/budget_gain.csv", index=False)
print("wrote results/budget_selected.csv, results/budget_gain.csv")
