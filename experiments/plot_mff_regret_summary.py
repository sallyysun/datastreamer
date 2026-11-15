"""Section 6.4 regret figures from the existing regret run (probe_strategies REGRET=1; analyze_regret.py): Gaussian copula,
KLL margins, log loss, eta = 1, mixture density forecast -log sum_k w_k c_k, five arms N_eff 39-1999, seeds 42-101.
Regret = cumulative MFF loss - comparator loss over the scored ticks (t >= 30; t >= 100 for the counting matrix).
  ch6_mff_regret_summary.png  final cumulative regret, fixed share (delta = 1e-4) only: static (best fixed arm) and
                              <= 1 switch (best arm sequence, exact DP); bars = mean over 60 seeds, whiskers ± 2 SE
                              (results/regret_seeds.csv). Constant and drift share a y-axis; the jump panel has its own.
  ch6_mff_regret_jump.png     abrupt jump only, n = 2000-5000: cumulative static and <= 1-switch regret, ordinary Hedge
                              (delta = 0) vs fixed share; the comparator re-optimised at each n (results/regret_paths.csv,
                              every 10 ticks); bands ± 2 SE.
Usage (repo root): python3 experiments/plot_mff_regret_summary.py -> images/ch6/6.4/"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
OUT = "images/ch6/6.4/"
os.makedirs(OUT, exist_ok=True)
EST = [("rmpl", "RMPL"), ("xiao", "Counting\nMatrix"), ("hermite", "Hermite\nSeries")]
EST1 = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
C_STAT, C_SW = "#3b4f7a", "#9fb0cf"
C_FS, C_H = "#0072B2", "#E69F00"
N = 60
S = pd.read_csv(core.R + "regret_seeds.csv")
P = pd.read_csv(core.R + "regret_paths.csv")
P = P[P.loss == "mix"]


def summary():
    fs = S[np.isclose(S.delta, 1e-4)]
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(core.W * 0.9, 2.45))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.32)
        axs = [fig.add_subplot(gs[0, 0])]
        axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0])); axs.append(fig.add_subplot(gs[0, 2]))
        bw = 0.34
        for ax, (sc, scl) in zip(axs, SC):
            for ei, (e, el) in enumerate(EST):
                g = fs[(fs.scenario == sc) & (fs.estimator == e)]
                for i, (x, col) in enumerate([(g.loss_mix - g.best0, C_STAT), (g.loss_mix - g.best1, C_SW)]):
                    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
                    xp = ei + (i - 0.5) * bw
                    ax.bar(xp, m, width=bw * 0.92, color=col, zorder=3)
                    ax.errorbar(xp, m, yerr=2 * se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
            ax.axhline(0, color="#222222", lw=0.8, zorder=2)
            ax.set_xticks(range(3)); ax.set_xticklabels([el for _, el in EST], fontsize=6.8); ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, 2.6)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.set_title(scl, fontsize=8, pad=3)
        axs[0].set_ylim(0, 5); axs[2].set_ylim(-19, 3)
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[0].set_ylabel("Final cumulative regret (nats)", fontsize=7)
        h = [plt.Rectangle((0, 0), 1, 1, color=C_STAT), plt.Rectangle((0, 0), 1, 1, color=C_SW)]
        fig.legend(h, ["Static regret", "Switching regret"],
                   loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.2, columnspacing=1.6)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.17)
        fig.savefig(OUT + "ch6_mff_regret_summary.png"); plt.close(fig)
    print("wrote", OUT + "ch6_mff_regret_summary.png")


def jump():
    T0 = 2000
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(core.W * 0.9, 3.3), sharex=True, sharey="row")
        for ci, (e, _) in enumerate(EST):
            for ri, m in enumerate([0, 1]):
                ax = axs[ri, ci]
                for d, col, ls in [(0.0, C_H, "-"), (1e-4, C_FS, "-")]:
                    g = P[(P.scenario == "jump") & (P.estimator == e) & np.isclose(P.delta, d) & (P.m == m) & (P.t >= T0)].sort_values("t")
                    band = 2 * g.sd / np.sqrt(N)
                    ax.fill_between(g.t, g["mean"] - band, g["mean"] + band, color=col, alpha=0.15, lw=0, zorder=2)
                    ax.plot(g.t, g["mean"], color=col, ls=ls, lw=1.3, zorder=3)
                ax.axhline(0, color="#b5b5b5", lw=0.5, zorder=1)
                ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                ax.set_xlim(T0, 5000); ax.set_xticks([2500, 3500, 4500])
                if ri == 0: ax.set_title(EST1[e], fontsize=8, pad=3)
        axs[0, 0].set_ylim(-18.5, 5); axs[1, 0].set_ylim(-4, 24)
        axs[0, 0].set_ylabel("Static regret\n(nats)", fontsize=7)
        axs[1, 0].set_ylabel("Switching regret,\n$\\leq$1 switch (nats)", fontsize=7)
        h = [plt.Line2D([], [], color=C_H, lw=1.3), plt.Line2D([], [], color=C_FS, lw=1.3)]
        fig.legend(h, [r"Ordinary Hedge ($\delta=0$)", r"Fixed share ($\delta=10^{-4}$)"], loc="upper center", ncol=2, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.55, 1.0), handlelength=2.4, columnspacing=1.6)
        fig.text(0.55, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.12, hspace=0.12, wspace=0.08)
        fig.savefig(OUT + "ch6_mff_regret_jump.png"); plt.close(fig)
    print("wrote", OUT + "ch6_mff_regret_jump.png")


summary()
jump()
fs = S[np.isclose(S.delta, 1e-4)]
for sc, _ in SC:
    for e, _ in EST:
        g = fs[(fs.scenario == sc) & (fs.estimator == e)]
        a, b = g.loss_mix - g.best0, g.loss_mix - g.best1
        print(f"{sc:9s} {e:8s} static {a.mean():7.2f} ± {2*a.std(ddof=1)/np.sqrt(len(a)):.2f}   switch1 {b.mean():6.2f} ± {2*b.std(ddof=1)/np.sqrt(len(b)):.2f}")
