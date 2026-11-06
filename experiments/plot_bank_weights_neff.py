# Bank weights through the jump: softmax only (plain Hedge) vs fixed share vs fixed share + cumulative
# arm. MPL arms, log loss, Gaussian, 60 seeds. Arms labelled by effective memory N_eff = (2 - g)/g.
# Reads results/bank_weights_paths[_<scenario>].csv (probe_bank_weights). See budget_mpl.md §16.
# Usage (from experiments/): python3 plot_bank_weights_neff.py [jump|drift]
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sc = sys.argv[1] if len(sys.argv) > 1 else "jump"
d = pd.read_csv(f"results/bank_weights_paths{'' if sc == 'jump' else '_' + sc}.csv")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEFF = [39, 99, 199, 399, 1999]
BLUES = ["#9ec5f4", "#6aa6ec", "#3987e5", "#1f63b8", "#103d73"]   # light = short memory, dark = long
TITLES = {"softmax": "(a) softmax only (Hedge, δ = 0)",
          "fixed_share": "(b) softmax + fixed share (δ = 3·10⁻⁴)",
          "fixed_share_cum": "(c) fixed share + cumulative arm"}
lo, hi = 1000, 4600

fig, ax = plt.subplots(3, 3, figsize=(13, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, 1]})
fig.patch.set_facecolor(SURF)
for j, v in enumerate(["softmax", "fixed_share", "fixed_share_cum"]):
    g = d[(d.variant == v) & (d.t >= lo) & (d.t <= hi)]
    w = g[g.k >= 0].pivot(index="t", columns="k", values="value")
    ne = g[g.k >= 0].pivot(index="t", columns="k", values="neff")
    for k in w.columns:
        if k < 5:
            ax[0, j].plot(w.index, w[k], color=BLUES[k], lw=1.8, label=f"N_eff = {NEFF[k]}")
        else:
            ax[0, j].plot(w.index, w[k], color=INK, lw=1.6, ls="--", label="cumulative (N_eff = t)")
    eff = (w * ne).sum(axis=1)
    ax[1, j].plot(eff.index, eff, color=INK, lw=1.6)
    se = g[g.k == -1].set_index("t").value
    rm = np.sqrt(se.rolling(10, center=True, min_periods=1).mean())
    ax[2, j].plot(rm.index, 1e3 * rm, color=INK, lw=1.4)
    ax[0, j].set_title(TITLES[v], fontsize=10.5, color=INK, loc="left")
    for i in range(3):
        a = ax[i, j]
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, lw=0.6)
        a.tick_params(labelsize=8.5, colors=MUTED)
        for s in a.spines.values():
            s.set_color(GRID)
        if sc == "jump":
            a.axvline(2500, color=MUTED, ls=":", lw=1)
    ax[0, j].set_ylim(-0.02, 1.02)
    ax[1, j].set_yscale("log")
    ax[2, j].set_yscale("log")
    ax[2, j].set_xlabel("tick", fontsize=9, color=MUTED)
ax[0, 0].set_ylabel("mean weight", fontsize=9, color=MUTED)
ax[1, 0].set_ylabel(r"bank memory $\sum_k w_k N_{\mathrm{eff},k}$" + "\n(observations, log)", fontsize=9, color=MUTED)
ax[2, 0].set_ylabel(r"RMSE of $\hat\rho_S$ ($\times10^{-3}$, log)", fontsize=9, color=MUTED)
for j in range(3):
    ax[1, j].set_ylim(25, 3500)
    ax[2, j].set_ylim(18, 700)
h, l = ax[0, 2].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
fig.suptitle(f"How the bank moves its weight: MPL log bank, Gaussian {sc} (ρ_S 0.19 → 0.79 at t = 2500), mean over 60 seeds"
             if sc == "jump" else f"How the bank moves its weight: MPL log bank, Gaussian drift, mean over 60 seeds",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = f"../notes/thesis/figures/ch6_bank_weights_neff{'' if sc == 'jump' else '_' + sc}.png"
fig.savefig(out, dpi=150)
print("wrote", out)
