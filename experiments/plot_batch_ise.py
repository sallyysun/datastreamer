"""rho_S RMSE and copula CDF ISE by scenario: family bank, Gaussian log bank (MPL), Hermite own-density bank, the
family bank's best fixed arm (hindsight), and the offline batch estimators (oracle segment, two-sided kernel with the
best bandwidth in hindsight). Geometric mean over the copula families of each scenario; 60 seeds.
Reads results/batch_ise_42*.csv (probe_batch_ise).
Usage (from experiments/): python3 plot_batch_ise.py -> ../notes/thesis/figures/ch6_batch_ise.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fams = ["gaussian", "clayton", "gumbel", "t", "gauss2t", "clayton2gumbel", "gauss2clayton"]
d = pd.concat(pd.read_csv(f"results/batch_ise_42{'' if c == 'gaussian' else '_' + c}.csv") for c in fams)
d = d[d.window == "whole"]
m = d.groupby(["copula", "scenario", "metric", "method"]).value.mean()

series = [("family_bank", "family bank (online)", "#2a78d6", "o", "-"),
          ("mle_bank_log", "Gaussian log bank, MPL (online)", "#eb6834", "s", "-"),
          ("hermite_bank_own", "Hermite own-density bank (online)", "#1baf7a", "^", "-"),
          ("best_arm", "best fixed arm, hindsight (online)", "#6250d6", "D", "-"),
          ("batch_segment", "batch, oracle segment (offline)", INK2, "x", "--"),
          ("batch_kernel", "batch, two-sided kernel, best h (offline)", INK, "+", "--")]
scen = ["constant", "drift", "jump", "oscillate", "shape"]


def value(metric, key, c, s):
    g = m.loc[(c, s, metric)]
    if key == "best_arm":
        return g[[i for i in g.index if i.startswith("arm_")]].min()
    if key == "batch_kernel":
        return g[[i for i in g.index if i.startswith("batch_kernel_")]].min()
    return g[key]


fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, metric, scale, title in [(ax[0], "rmse", 1000, r"(a) RMSE of $\rho_S$ ($\times10^{-3}$, log)"),
                                (ax[1], "ise_x1e4", 1, r"(b) Copula ISE $\int(\hat C-C)^2$ ($\times10^{-4}$, log)")]:
    for key, lab, col, mk, ls in series:
        ys = []
        for s in scen:
            cells = [(c, sc) for (c, sc) in m.index.droplevel([2, 3]).unique() if sc == s]
            ys.append(np.exp(np.mean([np.log(value(metric, key, c, sc) * scale) for c, sc in cells])))
        a.plot(range(len(scen)), ys, marker=mk, color=col, ls=ls, lw=1.4, ms=6, label=lab)
    a.set_yscale("log")
    a.set_xticks(range(len(scen)), ["constant", "drift", "jump", "oscillate", "shape switch"])
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title(title, loc="left", fontsize=9, color=INK)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.suptitle("Online banks against offline batch estimators, whole run, geometric mean over copula families, 60 seeds",
             fontsize=9, color=INK)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig("../notes/thesis/figures/ch6_batch_ise.png", dpi=170)
print("wrote ch6_batch_ise.png")
