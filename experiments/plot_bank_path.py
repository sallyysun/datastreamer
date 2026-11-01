"""Section 6.3 figure: one seed of the log-loss Xiao bank on drift and jump - its rho against the
truth and the fixed arms, its weights (stacked), and the weight-averaged memory.
Reads results/bank_path[_<family>].csv from probe_bank_path.
Usage (from experiments/): python3 plot_bank_path.py [family=gaussian]
  -> ../notes/thesis/figures/ch6_bank_path[_<family>].png"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fam = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
sfx = "" if fam == "gaussian" else "_" + fam
MEM = [39, 99, 199, 399, 1999]  # ForgettingBank arm order in registry.hpp
d = pd.read_csv(f"results/bank_path{sfx}.csv")
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(MEM)))
BANK = "#d95f02"

fig, ax = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1.3, 1]})
for j, sc in enumerate(["drift", "jump"]):
    g = d[d.scenario == sc]
    t = g.t.values
    a = ax[0, j]
    for k, m in enumerate(MEM):
        a.plot(t, g[f"arm_{k}"], color=blues[k], lw=0.8, label=f"fixed M = {m}")
    a.plot(t, g.truth, color="k", lw=1.2, ls="--", label=r"true $\rho_S$")
    a.plot(t, g.bank, color=BANK, lw=1.6, label="log-loss bank")
    a.set_title(sc)
    a.set_ylabel(r"Spearman $\rho$")
    w = np.vstack([g[f"w_{k}"].values for k in range(len(MEM))])
    ax[1, j].stackplot(t, w, colors=blues, edgecolor="white", linewidth=0.2,
                       labels=[f"M = {m}" for m in MEM])
    ax[1, j].set_ylim(0, 1)
    ax[1, j].set_ylabel("weight (stacked)")
    mem = (w * np.array(MEM)[:, None]).sum(axis=0)
    ax[2, j].plot(t, mem, color=BANK, lw=1.4)
    ax[2, j].set_yscale("log")
    ax[2, j].set_yticks(MEM)
    ax[2, j].set_yticklabels(MEM)
    ax[2, j].minorticks_off()
    ax[2, j].set_ylabel(r"$\sum_k w_k M_k$")
    ax[2, j].set_xlabel("tick")
    for r in range(3):
        ax[r, j].grid(alpha=0.25, lw=0.5)
        if sc == "jump":
            ax[r, j].axvline(2500, color="k", ls=":", lw=0.8)
ax[0, 0].legend(fontsize=8, loc="upper left")
h, l = ax[1, 1].get_legend_handles_labels()
ax[1, 1].legend(h[::-1], l[::-1], fontsize=7, loc="center right", framealpha=0.9)
fig.tight_layout()
os.makedirs("../notes/thesis/figures", exist_ok=True)
out = f"../notes/thesis/figures/ch6_bank_path{sfx}.png"
fig.savefig(out, dpi=150)
print(out)
