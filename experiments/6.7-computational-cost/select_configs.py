"""Section 6.7 validation-only configuration selection within each storage budget (n = 15,000): per family, scenario,
budget and estimator, the approximation size and gain with the lowest validation RMSE among those whose peak retained
state fits the budget, then its test RMSE; and the percentage RMSE reduction at 64 and 256 KiB relative to the 16 KiB
selection over the paired test seeds. Writes results/budget_selected.csv and results/budget_gain.csv."""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

EST = ["mpl_dskll", "xiao", "hermite"]


def select(sw, SD):
    val, tst = sw[sw.split == "val"], sw[sw.split == "test"]
    rows = []
    for f in C.FAMILIES_T31:
        for s in C.SCENARIOS:
            for b in C.BUDGETS:
                cap = b * 1024
                for e in EST:
                    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
                    r = v.loc[v.rmse.idxmin()]
                    a, g = int(r.approx), float(r.gain)
                    t = tst[(tst.family == f) & (tst.scenario == s) & (tst.estimator == e) & (tst.approx == a) & np.isclose(tst.gain, g)].iloc[0]
                    x = SD.loc[(f, s, e, a, g)]
                    rows.append(dict(family=f, scenario=s, budget_kib=b, estimator=e, approx=a, gain=g, val_rmse=100 * r.rmse,
                                     rmse=100 * x.mean(), se=100 * x.std(ddof=1) / math.sqrt(len(x)), peak_val=int(r.peak_bytes),
                                     peak_test=int(t.peak_bytes), cap=cap, util=100 * max(r.peak_bytes, t.peak_bytes) / cap))
    return pd.DataFrame(rows)


def gains(sel, SD):
    rows = []
    for (f, s, e), x in sel.groupby(["family", "scenario", "estimator"]):
        x = x.set_index("budget_kib")
        base = SD.loc[(f, s, e, int(x.loc[16].approx), float(x.loc[16].gain))].values
        for b in (64, 256):
            d = base - SD.loc[(f, s, e, int(x.loc[b].approx), float(x.loc[b].gain))].values
            rows.append(dict(family=f, scenario=s, estimator=e, budget=b, pct=100 * d.mean() / base.mean(),
                             pct_mc_se=100 * d.std(ddof=1) / np.sqrt(len(d)) / base.mean()))
    return pd.DataFrame(rows)


def main():
    sw, sd = pd.read_csv("results/budget_sweep_n15k.csv"), pd.read_csv("results/budget_seeds_n15k.csv")
    SD = sd.set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
    sel = select(sw, SD)
    sel.to_csv("results/budget_selected.csv", index=False)
    gains(sel, SD).to_csv("results/budget_gain.csv", index=False)
    print("wrote results/budget_selected.csv, results/budget_gain.csv")


if __name__ == "__main__":
    main()
