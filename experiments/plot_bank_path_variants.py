"""Single-seed bank path in the style of ch6_bank_path.png, for the spec MPL log bank, arms labelled by
effective memory N_eff = (2 - g)/g. Reads results/bank_path_variants.csv (probe_bank_path_variants).
  -> ../notes/thesis/figures/ch6_bank_path_mpl.png              fixed share (thesis bank), drift and jump
  -> ../notes/thesis/figures/ch6_bank_path_softmax_vs_fs.png    jump: softmax only vs softmax + fixed share
See notes/studies/budget_mpl.md §16. Usage (from experiments/): python3 plot_bank_path_variants.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEFF = [39, 99, 199, 399, 1999]
d = pd.read_csv("results/bank_path_variants.csv")
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(NEFF)))
BANK = "#d95f02"


def column(ax, g, title, jump):
    t = g.t.values
    a = ax[0]
    for k, n in enumerate(NEFF):
        a.plot(t, g[f"arm_{k}"], color=blues[k], lw=0.8, label=f"fixed N_eff = {n}")
    a.plot(t, g.truth, color="k", lw=1.2, ls="--", label=r"true $\rho_S$")
    a.plot(t, g.bank, color=BANK, lw=1.6, label="log-loss bank")
    a.set_title(title)
    a.set_ylabel(r"Spearman $\rho$")
    w = np.vstack([g[f"w_{k}"].values for k in range(len(NEFF))])
    ax[1].stackplot(t, w, colors=blues, edgecolor="white", linewidth=0.2, labels=[f"N_eff = {n}" for n in NEFF])
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("weight (stacked)")
    mem = (w * np.array(NEFF)[:, None]).sum(axis=0)
    ax[2].plot(t, mem, color=BANK, lw=1.4)
    ax[2].set_yscale("log")
    ax[2].set_yticks(NEFF)
    ax[2].set_yticklabels(NEFF)
    ax[2].minorticks_off()
    ax[2].set_ylim(35, 2200)
    ax[2].set_ylabel(r"$\sum_k w_k N_{\mathrm{eff},k}$")
    ax[2].set_xlabel("tick")
    for r in range(3):
        ax[r].grid(alpha=0.25, lw=0.5)
        if jump:
            ax[r].axvline(2500, color="k", ls=":", lw=0.8)


def figure(cols, out, suptitle):
    fig, ax = plt.subplots(3, len(cols), figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1]})
    for j, (variant, sc, title) in enumerate(cols):
        column(ax[:, j], d[(d.variant == variant) & (d.scenario == sc)], title, sc == "jump")
        if sc == "jump":
            ax[0, j].set_ylim(-0.45, 0.95)
    ax[0, 0].legend(fontsize=8, loc="upper left")
    h, l = ax[1, -1].get_legend_handles_labels()
    ax[1, -1].legend(h[::-1], l[::-1], fontsize=7, loc="center right", framealpha=0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


figure([("fixed_share", "drift", "drift"), ("fixed_share", "jump", "jump")],
       "../notes/thesis/figures/ch6_bank_path_mpl.png", None)
figure([("softmax", "jump", "softmax only (Hedge, δ = 0)"), ("fixed_share", "jump", "softmax + fixed share (δ = 3·10⁻⁴)")],
       "../notes/thesis/figures/ch6_bank_path_softmax_vs_fs.png", "MPL log bank on jump, one seed (42)")
