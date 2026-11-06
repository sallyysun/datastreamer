# Kalman / BOCPD methods vs the regenerated Chapter 6 log banks, same streams (seeds 42-101), RMSE x1e-3.
# Paired differences vs the MPL log bank: mean ± 2 SE, * = beyond 2 SE, negative = method better.
import glob
import numpy as np
import pandas as pd

S = ["", "_clayton", "_gumbel", "_t"]
k = pd.concat(pd.read_csv(f) for f in glob.glob("results/kalman_ch6*.csv"))
b = pd.concat(pd.read_csv(f"results/thesis_bank{s}.csv") for s in S)
b = b[b.method == "bank_log"].assign(method=lambda x: x.component + "_bank_log")[["copula", "scenario", "seed", "window", "method", "rmse"]]
d = pd.concat([k, b])
M = ["xiao_bank_log", "mle_bank_log", "kalman", "kalman_fast", "kalman_vff", "kalman_bank", "kalman_bank_sticky",
     "kalman_bocpd", "kalman_bocpd_fast", "kalman_cusum", "kalman_glr"]
pd.set_option("display.width", 250)
for win, scs in [("whole", ["constant", "drift", "jump", "oscillate"]), ("post", ["jump"]), ("rec_0_100", ["jump"])]:
    print(f"\n== mean RMSE x1e-3, window {win} (rows scenario x family) ==")
    t = d[d.window == win].groupby(["scenario", "copula", "method"]).rmse.mean().unstack()[M] * 1e3
    print(t.loc[scs].round(1).to_string())
print("\n== paired vs MPL log bank, whole run, mean over the 4 families of (method - mle_bank_log), x1e-3 ==")
p = d[d.window == "whole"].pivot_table(index=["scenario", "copula", "seed"], columns="method", values="rmse")
for sc in ["constant", "drift", "jump", "oscillate"]:
    g = p.loc[sc]
    row = []
    for m in M:
        x = (g[m] - g["mle_bank_log"]) * 1e3
        row.append(f"{m}: {x.mean():+.1f}±{2 * x.std(ddof=1) / np.sqrt(len(x)):.1f}")
    print(sc, " | ".join(row[2:]))
