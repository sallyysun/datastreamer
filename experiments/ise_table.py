# Whole-copula accuracy of Xiao / Hermite / MPL (budget_mpl.md §18) from results/ise*.csv (probe_ise).
# Density ISE on the normal-score scale and copula CDF ISE, whole run (and post-change for jump/shape).
# Paired differences mean ± 2 SE over seeds; * = beyond 2 SE.
import glob
import numpy as np
import pandas as pd

d = pd.concat(pd.read_csv(f) for f in glob.glob("results/ise*.csv"))
CELLS = [(f, s) for f in ["gaussian", "clayton", "gumbel", "t"] for s in ["constant", "drift", "jump"]] + \
        [("clayton2gumbel", "shape"), ("gauss2t", "shape")]
C = ["xiao", "hermite", "mle"]
pd.set_option("display.width", 250)


def cell(x):
    m, h = x.mean(), 2 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+.2e}±{h:.1e}{'*' if abs(m) > h else ' '}"


for metric, unit, sc_ in [("ise_density", "x1e-3", 1e3), ("ise_cdf", "x1e-4", 1e4)]:
    for win in ["whole", "post"]:
        rows = []
        for f, s in CELLS:
            if win == "post" and s not in ("jump", "shape"):
                continue
            g = d[(d.copula == f) & (d.scenario == s) & (d.window == win)]
            p = g.pivot_table(index="seed", columns="method", values=metric)
            r = {"cell": f"{f}·{s}"}
            for c in C:
                fixed = [f"{c}_neff{n}" for n in [39, 99, 199, 399, 1999]]
                best = p[fixed].mean().idxmin()
                r[f"{c}_bank_own"] = sc_ * p[f"{c}_bank_own"].mean()
                r[f"{c}_best_fixed"] = f"{sc_ * p[best].mean():.3f} ({best.split('neff')[1]})"
            r["gauss_at_rho(mle log bank)"] = sc_ * p["mle_bank_log_gauss"].mean()
            r["gauss_at_rho(xiao log bank)"] = sc_ * p["xiao_bank_log_gauss"].mean()
            own = {c: p[f"{c}_bank_own"].mean() for c in C}
            r["best_own_bank"] = min(own, key=own.get)
            rows.append(r)
        print(f"\n== {metric} ({unit}), window {win} ==")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
