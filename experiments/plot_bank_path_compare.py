"""Stacked-weight comparison of the Xiao, Hermite and MPL log banks on one seed, in the style of
ch6_bank_path.png: estimates, stacked weights and bank memory sum_k w_k N_eff,k. All three banks have
the same five memories, N_eff = 39 ... 1999. Reads results/bank_path_{xiao,hermite,mle}.csv
(./probe_bank_path 42 <component>). Usage (from experiments/): python3 plot_bank_path_compare.py [jump|drift]
  -> ../notes/thesis/figures/ch6_bank_path_compare_<scenario>.png"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sc = sys.argv[1] if len(sys.argv) > 1 else "jump"
NEFF = [39, 99, 199, 399, 1999]
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(NEFF)))
BANK = "#d95f02"
COMP = [("xiao", "Xiao bank"), ("hermite", "Hermite bank"), ("mle", "MPL bank")]

fig, ax = plt.subplots(3, 3, figsize=(15, 7.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1]})
for j, (c, title) in enumerate(COMP):
    g = pd.read_csv(f"results/bank_path_{c}.csv")
    g = g[g.scenario == sc]
    t = g.t.values
    a = ax[0, j]
    for k, n in enumerate(NEFF):
        a.plot(t, g[f"arm_{k}"], color=blues[k], lw=0.8, label=f"fixed N_eff = {n}")
    a.plot(t, g.truth, color="k", lw=1.2, ls="--", label=r"true $\rho_S$")
    a.plot(t, g.bank, color=BANK, lw=1.6, label="log-loss bank")
    rmse = 1e3 * np.sqrt(((g.bank - g.truth) ** 2)[g.t >= 500].mean())
    a.set_title(f"{title}   (RMSE {rmse:.0f} ×10⁻³, this seed)")
    a.set_ylim(-0.75, 1.0)
    w = np.vstack([g[f"w_{k}"].values for k in range(len(NEFF))])
    ax[1, j].stackplot(t, w, colors=blues, edgecolor="white", linewidth=0.2, labels=[f"N_eff = {n}" for n in NEFF])
    ax[1, j].set_ylim(0, 1)
    mem = (w * np.array(NEFF)[:, None]).sum(axis=0)
    ax[2, j].plot(t, mem, color=BANK, lw=1.4)
    ax[2, j].set_yscale("log")
    ax[2, j].set_yticks(NEFF)
    ax[2, j].set_yticklabels(NEFF)
    ax[2, j].minorticks_off()
    ax[2, j].set_ylim(35, 2200)
    ax[2, j].set_xlabel("tick")
    for r in range(3):
        ax[r, j].grid(alpha=0.25, lw=0.5)
        if sc == "jump":
            ax[r, j].axvline(2500, color="k", ls=":", lw=0.8)
ax[0, 0].set_ylabel(r"Spearman $\rho$")
ax[1, 0].set_ylabel("weight (stacked)")
ax[2, 0].set_ylabel(r"$\sum_k w_k N_{\mathrm{eff},k}$")
ax[0, 0].legend(fontsize=8, loc="upper left")
h, l = ax[1, 2].get_legend_handles_labels()
ax[1, 2].legend(h[::-1], l[::-1], fontsize=7, loc="center right", framealpha=0.9)
fig.suptitle(f"Log-loss banks on {sc}, one seed (42), Gaussian copula: same five memories, three estimators", fontsize=11)
fig.tight_layout()
out = f"../notes/thesis/figures/ch6_bank_path_compare_{sc}.png"
fig.savefig(out, dpi=150)
print(out)
