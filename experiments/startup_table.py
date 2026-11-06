# MPL-bank start-up comparison (budget_mpl.md §12): (b) mle = constant gain with max(gain, 1/m) start,
# (a) mlea = constant gain from the first update, mleold = the previous ParametricCopulaMLE arms.
# Reads results/thesis_bank_startup[_fam].csv; differences are paired over seeds, mean ± 2 SE, x1e-3.
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
d = pd.concat(pd.read_csv(f) for f in glob.glob("results/thesis_bank_startup*.csv"))
key = ["copula", "scenario", "seed", "window", "method"]
w = d.pivot_table(index=key, columns="component", values="rmse").reset_index()
FAM = ["gaussian", "clayton", "gumbel", "t"]


def cell(x):
    m, s = 1e3 * x.mean(), 2e3 * x.std() / np.sqrt(len(x))
    flag = "*" if abs(m) > s else " "
    return f"{m:+6.2f}±{s:4.2f}{flag}"


def means(sc, win, meth):
    r = w[(w.scenario == sc) & (w.window == win) & (w.method == meth)]
    return r.groupby("copula")[["mle", "mlea", "mleold"]].mean().reindex(FAM) * 1e3


print("== sanity: mleold log bank, whole run (x1e-3); chapter6_results.md has Gaussian 67.0 drift, 59.6 jump ==")
for sc in ["drift", "jump"]:
    print(sc, means(sc, "whole", "bank_log").mleold.round(1).to_dict())

rows = []
for sc, win in [("drift", "whole"), ("drift", "early"), ("jump", "whole"), ("jump", "early"), ("jump", "post"),
                ("jump", "rec_0_100"), ("constant", "whole"), ("constant", "early")]:
    for meth in ["bank_log", "bank", "fixed_1999", "fixed_199"]:
        for fam in FAM:
            r = w[(w.copula == fam) & (w.scenario == sc) & (w.window == win) & (w.method == meth)]
            rows.append(dict(scenario=sc, window=win, method=meth, copula=fam,
                             b=1e3 * r.mle.mean(), a=1e3 * r.mlea.mean(), old=1e3 * r.mleold.mean(),
                             b_old=cell(r.mle - r.mleold), a_old=cell(r.mlea - r.mleold), b_a=cell(r.mle - r.mlea)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print("\n== RMSE (x1e-3) and paired differences, mean ± 2 SE (* = beyond 2 SE) ==")
print(t.round(1).to_string(index=False))

# Bank vs fixed arms under (b), as in chapter6_results.md, drift + jump
print("\n== (b): log bank minus fixed arms, whole run (x1e-3, paired) ==")
out = []
for sc in ["drift", "jump"]:
    for fam in FAM:
        r = d[(d.copula == fam) & (d.scenario == sc) & (d.window == "whole") & (d.component == "mle")]
        p = r.pivot_table(index="seed", columns="method", values="rmse")
        fixed = [c for c in p.columns if c.startswith("fixed_")]
        best = p[fixed].mean().idxmin()
        out.append(dict(scenario=sc, copula=fam, best_fixed=best, log_bank=1e3 * p.bank_log.mean(),
                        minus_best=cell(p.bank_log - p[best]), minus_M199=cell(p.bank_log - p.fixed_199),
                        sq_minus_best=cell(p.bank - p[best])))
print(pd.DataFrame(out).round(1).to_string(index=False))
