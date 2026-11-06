# Cumulative arm in the bank (budget_mpl.md §15): <est>cum bank (5 forgetting arms + 1 cumulative, K = 6)
# against the 5-arm bank and the cumulative estimator alone. Paired over seeds 42-101, RMSE x1e-3,
# mean ± 2 SE, * = beyond 2 SE, negative = with-cum better.
import glob
import numpy as np
import pandas as pd

d = pd.concat(pd.read_csv(f) for f in glob.glob("results/noforget*.csv"))
F = ["gaussian", "clayton", "gumbel", "t"]


def cell(x):
    m, h = 1e3 * x.mean(), 2e3 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+6.1f}±{h:4.1f}{'*' if abs(m) > h else ' '}"


for est, loss in [("xiao", "log"), ("hermite", "log"), ("mpl", "log"), ("xiao", "sq"), ("mpl", "sq")]:
    print(f"\n== {est}, {loss} bank: RMSE x1e-3 ==")
    rows = []
    for sc, win in [("constant", "whole"), ("drift", "whole"), ("jump", "whole"), ("jump", "early"),
                    ("jump", "post"), ("oscillate", "whole")]:
        for fam in F:
            p = d[(d.copula == fam) & (d.scenario == sc) & (d.window == win)].pivot_table(index="seed", columns="method", values="rmse")
            a, b, c = p[f"{est}_bankcum_{loss}"], p[f"{est}_bank_{loss}"], p[f"{est}_cum"]
            r = dict(scenario=sc, window=win, copula=fam, cum=1e3 * c.mean(), bank5=1e3 * b.mean(), bank6=1e3 * a.mean(),
                     bank6_minus_bank5=cell(a - b), bank6_minus_cum=cell(a - c))
            if loss == "log" and est in ("xiao", "mpl"):
                w = d[(d.copula == fam) & (d.scenario == sc) & (d.window == ("post" if win == "post" else "whole")) & (d.method == f"w_cum_{est}")]
                r["w_cum"] = w.rmse.mean()
            rows.append(r)
    pd.set_option("display.width", 230)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
