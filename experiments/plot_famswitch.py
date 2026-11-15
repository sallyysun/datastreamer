"""Section 6.5 figures from existing family-bank outputs (probe_family_bank, TAG density3b for the shape switches,
density3 for the Gaussian constant baseline; EW margins N_eff,m = 500; seeds 42-101 in four 15-seed chunks).
  ch6_famswitch_weights.png   summed weight on the original and the new family (over memories 99/399/1999) per tick;
                              the 60-seed mean = mean of the four 15-seed chunk means (equal sizes). Seed-level paths
                              were not saved, so no SE bands are drawn.
  ch6_famswitch_density.png   post-switch (ticks 2500-4999) excess log loss, family-augmented MFF (family bank) vs the
                              Gaussian-only bank with the same machinery (gauss_bank), mean ± 2 SE over 60 seeds
                              (results/famswitch_density.csv from analyze_famswitch.py); plus the Gaussian constant path
                              (ticks 500-4999) as the cost of family augmentation under a correct Gaussian model.
Usage (repo root): python3 experiments/plot_famswitch.py -> images/ch6/6.5/"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
OUT = "images/ch6/6.5/"
os.makedirs(OUT, exist_ok=True)
REG = [("gauss2t", r"Gaussian $\rightarrow$ $t_4$", "gaussian", "t"),
       ("gauss2clayton", r"Gaussian $\rightarrow$ Clayton", "gaussian", "clayton"),
       ("clayton2gumbel", r"Clayton $\rightarrow$ Gumbel", "clayton", "gumbel")]
FL = {"gaussian": "Gaussian", "t": "$t_4$", "clayton": "Clayton", "gumbel": "Gumbel"}
C_OLD, C_NEW = "#7a7a7a", "#0072B2"
C_FAM, C_GAU = "#C2185B", "#9a9a9a"


def weights():
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.5), sharey=True)
        for ax, (reg, rl, old, new) in zip(axs, REG):
            p = pd.concat([pd.read_csv(core.R + f"family_bank_path_{s}_density3b_{reg}.csv") for s in (42, 57, 72, 87)])
            p = p[p.scenario == "shape"]
            for fam, col, lab in [(old, C_OLD, "original"), (new, C_NEW, "new")]:
                g = p[p.series == f"w_{fam}"].groupby("t").value.mean()
                g = g[g.index >= 500]
                ax.plot(g.index, g.values, color=col, lw=1.3, zorder=3, label=f"{FL[fam]} ({lab})")
            ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=2)
            ax.axhline(0.5, color="#b5b5b5", lw=0.6, zorder=1)
            ax.set_xlim(500, 5000); ax.set_xticks([1000, 2500, 4000]); ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.set_title(rl, fontsize=8, pad=3)
        axs[0].set_ylabel("Summed family weight", fontsize=7.5)
        h = [plt.Line2D([], [], color=C_OLD, lw=1.3), plt.Line2D([], [], color=C_NEW, lw=1.3)]
        fig.legend(h, ["Original family", "New family"], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.54, 1.0), handlelength=2.0, columnspacing=1.6)
        fig.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.18, wspace=0.08)
        fig.savefig(OUT + "ch6_famswitch_weights_lines.png"); plt.close(fig)
    print("wrote", OUT + "ch6_famswitch_weights_lines.png")


def density():
    """Reduction in post-switch (ticks 2500-4999) excess log loss, Delta L = L(Gaussian-only MFF) - L(family-augmented
    MFF), mean ± 2 SE over 60 paired replications (results/famswitch_final.csv from famswitch_final.py). Linear axis."""
    D = pd.read_csv(core.R + "famswitch_final.csv")
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(core.W * 0.6, 2.3))
        for i, (reg, rl, _, _) in enumerate(REG):
            r = D[(D.regime == reg) & (D.window == "post")].iloc[0]
            ax.bar(i, r.dL, width=0.55, color=C_FAM, zorder=3)
            ax.errorbar(i, r.dL, yerr=r.dL_2se, color="#3a3a3a", lw=0.9, capsize=0, zorder=4)
        ax.axhline(0, color="#222222", lw=0.7, zorder=2)
        ax.set_xticks(range(len(REG))); ax.set_xticklabels([rl.replace(r" $\rightarrow$ ", " →\n") for _, rl, _, _ in REG], fontsize=7)
        ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(REG) - 0.4); ax.set_ylim(0, 80)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_ylabel("Reduction in excess log loss\n($\\times10^{-3}$ nats per observation)", fontsize=7)
        fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.2)
        fig.savefig(OUT + "ch6_famswitch_density.png"); plt.close(fig)
    print("wrote", OUT + "ch6_famswitch_density.png")


def weights_stacked():
    """Stacked seed-mean family weights (summed over memories): Gaussian, t4, Clayton, Gumbel, and the four rotated shapes
    together; renormalised per tick (the seed-mean shares sum to 1 up to print rounding). -> ch6_famswitch_weights.png (the main figure; the line version is ch6_famswitch_weights_lines.png)"""
    FAMS = [("w_gaussian", "Gaussian", "#9a9a9a"), ("w_t", "$t_4$", "#0072B2"), ("w_clayton", "Clayton", "#E69F00"),
            ("w_gumbel", "Gumbel", "#009E73"), ("rot", "Rotated Clayton / Gumbel", "#CC79A7")]
    ROT = ["w_clayton90", "w_clayton270", "w_gumbel90", "w_gumbel270"]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.45), sharey=True)
        for ax, (reg, rl, old, new) in zip(axs, REG):
            p = pd.concat([pd.read_csv(core.R + f"family_bank_path_{s}_density3b_{reg}.csv") for s in (42, 57, 72, 87)])
            p = p[(p.scenario == "shape") & p.series.str.startswith("w_")]
            W = p.groupby(["t", "series"]).value.mean().unstack()
            W = W[W.index >= 500]
            W["rot"] = W[ROT].sum(axis=1)
            M = W[[k for k, _, _ in FAMS]]
            M = M.div(M.sum(axis=1), axis=0)
            ax.stackplot(M.index, *[M[k] for k, _, _ in FAMS], colors=[c for _, _, c in FAMS], lw=0, zorder=2)
            ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
            ax.set_xlim(500, 5000); ax.set_xticks([1000, 2500, 4000]); ax.set_ylim(0, 1)
            ax.set_title(rl, fontsize=8, pad=3)
        axs[0].set_ylabel("Mean MFF weight", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in FAMS]
        fig.legend(h, [l for _, l, _ in FAMS], loc="upper center", ncol=5, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.54, 1.0), handlelength=1.2, columnspacing=1.2)
        fig.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.18, wspace=0.08)
        fig.savefig(OUT + "ch6_famswitch_weights.png"); plt.close(fig)   # main Figure (stacked)
    print("wrote", OUT + "ch6_famswitch_weights.png")


weights()
density()
weights_stacked()
