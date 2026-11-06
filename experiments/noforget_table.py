# Forgetting vs no forgetting (budget_mpl.md §14): log bank and M = 199 against the cumulative estimator,
# paired over seeds 42-101. RMSE x1e-3; differences mean ± 2 SE; * = beyond 2 SE; negative = forgetting better.
import glob
import numpy as np
import pandas as pd

d = pd.concat(pd.read_csv(f) for f in glob.glob("results/noforget*.csv"))
F = ["gaussian", "clayton", "gumbel", "t"]


def cell(x):
    m, h = 1e3 * x.mean(), 2e3 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+6.1f}±{h:4.1f}{'*' if abs(m) > h else ' '}"


print("== sanity: xiao_bank_log whole, Gaussian drift/jump (chapter6_results.md: 70.6 / 62.0) ==")
for sc in ["drift", "jump"]:
    print(sc, round(1e3 * d[(d.copula == "gaussian") & (d.scenario == sc) & (d.window == "whole") & (d.method == "xiao_bank_log")].rmse.mean(), 1))
for est in ["xiao", "hermite", "mpl"]:
    print(f"\n== {est}: RMSE x1e-3; bank − no-forgetting, M199 − no-forgetting ==")
    rows = []
    for sc, win in [("constant", "whole"), ("drift", "whole"), ("jump", "whole"), ("jump", "early"), ("jump", "post"), ("oscillate", "whole")]:
        for fam in F:
            p = d[(d.copula == fam) & (d.scenario == sc) & (d.window == win)].pivot_table(index="seed", columns="method", values="rmse")
            r = dict(scenario=sc, window=win, copula=fam, cum=1e3 * p[f"{est}_cum"].mean(),
                     m199=1e3 * p[f"{est}_m199"].mean(), bank=1e3 * p[f"{est}_bank_log"].mean(),
                     bank_minus_cum=cell(p[f"{est}_bank_log"] - p[f"{est}_cum"]),
                     m199_minus_cum=cell(p[f"{est}_m199"] - p[f"{est}_cum"]))
            if est == "xiao":
                r["raw_cum"] = 1e3 * p["xiao_raw_cum"].mean()
                r["bank_minus_raw"] = cell(p["xiao_bank_log"] - p["xiao_raw_cum"])
            rows.append(r)
    pd.set_option("display.width", 230)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
