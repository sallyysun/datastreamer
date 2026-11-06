# Mixed bank (5 Xiao + 5 Hermite + 5 spec-MPL arms) against the single-estimator banks, same seeds and
# streams (budget_mpl.md §13). Reads results/thesis_bank[_fam].csv and results/thesis_bank_mixed[_fam].csv.
# Paired differences x1e-3, mean ± 2 SE; * = beyond 2 SE; negative = mixed better.
import numpy as np
import pandas as pd

S = ["", "_clayton", "_gumbel", "_t"]
d = pd.concat([pd.read_csv(f"results/thesis_bank{s}.csv") for s in S] +
              [pd.read_csv(f"results/thesis_bank_mixed{s}.csv") for s in S])
d = d[~((d.component == "mixed") & d.method.str.startswith("fixed_"))]   # its first 5 arms are Xiao's


def cell(x):
    m, h = 1e3 * x.mean(), 2e3 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+5.1f}±{h:3.1f}{'*' if abs(m) > h else ' '}"


for meth in ["bank_log", "bank"]:
    print(f"\n== {meth}: RMSE x1e-3 (mixed, and mixed minus each single bank) ==")
    rows = []
    for sc, win in [("drift", "whole"), ("jump", "whole"), ("jump", "post"), ("jump", "rec_0_100"),
                    ("constant", "whole"), ("oscillate", "whole")]:
        for fam in ["gaussian", "clayton", "gumbel", "t"]:
            x = d[(d.copula == fam) & (d.scenario == sc) & (d.window == win) & (d.method == meth)]
            p = x.pivot_table(index="seed", columns="component", values="rmse")
            best = min(["xiao", "hermite", "mle"], key=lambda c: p[c].mean())
            rows.append(dict(scenario=sc, window=win, copula=fam, mixed=1e3 * p.mixed.mean(),
                             xiao=1e3 * p.xiao.mean(), hermite=1e3 * p.hermite.mean(), mle=1e3 * p.mle.mean(),
                             m_minus_xiao=cell(p.mixed - p.xiao), m_minus_mle=cell(p.mixed - p.mle),
                             m_minus_best=cell(p.mixed - p[best]), best=best))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.round(1).to_string(index=False))
