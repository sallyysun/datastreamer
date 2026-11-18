"""Section 6.4 figures: log-loss gains by generating and working family, density bars, and MFF regret."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import format_estimate, verbose_print  # noqa: E402
from strategy_allfixed import plot_all_fixed  # noqa: E402
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
FAM = C.FAMILIES_T31
FL = [C.FAMILY_MATH[f] for f in FAM]
MET = [("rmse", "Spearman RMSE"), ("h2_x1000", "Squared Hellinger"), ("excess_x1000", "Excess log loss")]
LIM = 60
MINUS = "−"
C_STAT, C_SW = ce.REGRET["static"], ce.REGRET["switching"]
C_FS, C_H = ce.FIXED_SHARE, ce.HEDGE
N = len(C.SEEDS)


def gain_matrix(cells, sc, e, m):
    A, S = np.full((4, 4), np.nan), np.zeros((4, 4), bool)
    for i, g in enumerate(FAM):
        for j, w in enumerate(FAM):
            r = cells[(cells.gen == g) & (cells.working == w) & (cells.scenario == sc) & (cells.estimator == e) & (cells.metric == m)].iloc[0]
            A[i, j] = 100 * r["diff"] / r["sq"]
            S[i, j] = r["verdict"] != "tie"
    return A, S


def heatmap_panel(ax, A, S, ri, ci):
    im = ax.imshow(np.clip(A, -LIM, LIM), cmap="RdBu", vmin=-LIM, vmax=LIM)
    for i in range(4):
        for j in range(4):
            strong = abs(np.clip(A[i, j], -LIM, LIM)) > 38
            col = ("white" if strong else ce.DARK) if S[i, j] else ce.REF
            ax.text(j, i, f"{A[i, j]:+.0f}".replace("-", MINUS), ha="center", va="center", fontsize=7.2,
                    color=col, fontweight=("bold" if i == j else "normal"))
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(FL if ri == 2 else [], fontsize=7.2, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(FL if ci == 0 else [], fontsize=7.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=2)
    return im


def colourbar(f, im):
    cax = f.add_axes([0.3, 0.078, 0.5, 0.016])
    cb = f.colorbar(im, cax=cax, orientation="horizontal", ticks=[-60, -30, 0, 30, 60])
    cb.ax.set_xticklabels([MINUS + "60", MINUS + "30", "0\n(no difference)", "30", "60"], fontsize=6.8)
    cb.ax.tick_params(length=2, pad=1.5)
    cb.set_label("Relative error reduction with logarithmic loss (%)", fontsize=7.5, labelpad=2)
    cax.text(-0.02, 0.5, "Squared loss\nbetter", transform=cax.transAxes, ha="right", va="center", fontsize=7, color=ce.SQUARED_BETTER)
    cax.text(1.02, 0.5, "Log loss\nbetter", transform=cax.transAxes, ha="left", va="center", fontsize=7, color=ce.LOG_BETTER)


def plot_scoresfam_heatmap(e):
    """Log-loss gain over squared loss by generating x working family, one 3 x 3 grid per estimator."""
    cells = pd.read_csv(ce.R + "scoresfam_cells.csv")
    cells = cells[cells.window == "whole"]
    with plt.rc_context(ce.RC):
        f, axs = plt.subplots(3, 3, figsize=(ce.W, 6.0))
        for ri, (m, ml) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                im = heatmap_panel(axs[ri, ci], *gain_matrix(cells, sc, e, m), ri, ci)
                if ri == 0:
                    axs[ri, ci].set_title(scl, fontsize=8.5, pad=4)
            axs[ri, 0].annotate(ml, xy=(0, 0.5), xycoords="axes fraction", xytext=(-44, 0), textcoords="offset points",
                                rotation=90, ha="center", va="center", fontsize=8)
        f.text(0.03, 0.6, "Generating family", rotation=90, ha="center", va="center", fontsize=8)
        ce.bottom_label(f, "Estimation family", x=0.555, y=0.142)
        colourbar(f, im)
        f.subplots_adjust(left=0.2, right=0.99, top=0.96, bottom=0.225, hspace=0.07, wspace=0.05)
        ce.save(f, f"ch6_scoresfam_heatmap_{e}.png")


def plot_density_bars():
    plot_all_fixed(Path("."), "density")


def fixed_share_regret():
    S = pd.read_csv(ce.R + "regret_seeds.csv")
    return S[np.isclose(S.delta, 1e-4)]


def plot_regret_summary():
    """Final static and one-switch regret of fixed share, mean ± 2 SE."""
    fs = fixed_share_regret()
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(ce.W * 0.9, 2.45))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.32)
        axs = [fig.add_subplot(gs[0, 0])]
        axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))
        axs.append(fig.add_subplot(gs[0, 2]))
        bw = 0.34
        for ax, (sc, scl) in zip(axs, SC):
            for ei, e in enumerate(C.ESTIMATORS):
                g = fs[(fs.scenario == sc) & (fs.estimator == e)]
                for i, (x, col) in enumerate([(g.loss_mix - g.best0, C_STAT), (g.loss_mix - g.best1, C_SW)]):
                    m, se = x.mean(), standard_error(x)
                    xp = ei + (i - 0.5) * bw
                    ax.bar(xp, m, width=bw * 0.92, color=col, zorder=3)
                    ax.errorbar(xp, m, yerr=2 * se, color=ce.EDGE, lw=0.8, capsize=0, zorder=4)
            ax.axhline(0, color=ce.DARK, lw=0.8, zorder=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels([C.EST_LABEL2[e] for e in C.ESTIMATORS], fontsize=6.8)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, 2.6)
            ce.grid(ax, "y")
            ax.set_title(scl, fontsize=8, pad=3)
        axs[0].set_ylim(0, 5)
        axs[2].set_ylim(-19, 3)
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[0].set_ylabel("Final cumulative regret (nats)", fontsize=7)
        h = [ce.swatch(color=C_STAT), ce.swatch(color=C_SW)]
        ce.top_legend(fig, h, ["Static regret", "Switching regret"], ncol=2, handlelength=1.2, columnspacing=1.6, x=0.54)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.17)
        ce.save(fig, "ch6_mff_regret_summary.png")


def plot_regret_jump(t0=2000):
    """Cumulative regret through the jump, Hedge vs fixed share, ± 2 SE bands."""
    P = pd.read_csv(ce.R + "regret_paths.csv")
    P = P[P.loss == "mix"]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(ce.W * 0.9, 3.3), sharex=True, sharey="row")
        for ci, e in enumerate(C.ESTIMATORS):
            for ri, m in enumerate([0, 1]):
                ax = axs[ri, ci]
                for d, col in [(0.0, C_H), (1e-4, C_FS)]:
                    g = P[(P.scenario == "jump") & (P.estimator == e) & np.isclose(P.delta, d) & (P.m == m) & (P.t >= t0)].sort_values("t")
                    band = 2 * g.sd / np.sqrt(N)
                    ax.fill_between(g.t, g["mean"] - band, g["mean"] + band, color=col, alpha=0.15, lw=0, zorder=2)
                    ax.plot(g.t, g["mean"], color=col, ls="-", lw=1.3, zorder=3)
                ax.axhline(0, color=ce.REF, lw=0.5, zorder=1)
                ax.axvline(C.CHANGE, color=ce.DARK, lw=0.8, ls=(0, (4, 3)), zorder=4)
                ce.grid(ax, "y")
                ax.set_xlim(t0, C.T)
                ax.set_xticks([2500, 3500, 4500])
                if ri == 0:
                    ax.set_title(C.EST_LABEL[e], fontsize=8, pad=3)
        axs[0, 0].set_ylim(-18.5, 5)
        axs[1, 0].set_ylim(-4, 24)
        axs[0, 0].set_ylabel("Static regret\n(nats)", fontsize=7)
        axs[1, 0].set_ylabel("Switching regret,\n$\\leq$1 switch (nats)", fontsize=7)
        h = [plt.Line2D([], [], color=C_H, lw=1.3), plt.Line2D([], [], color=C_FS, lw=1.3)]
        ce.top_legend(fig, h, [r"Ordinary Hedge ($\delta=0$)", r"Fixed share ($\delta=10^{-4}$)"], ncol=2, handlelength=2.4, columnspacing=1.6, x=0.55)
        ce.bottom_label(fig, "Observation number, $n$", x=0.55, fontsize=7.5)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.12, hspace=0.12, wspace=0.08)
        ce.save(fig, "ch6_mff_regret_jump.png")


def print_regret():
    fs = fixed_share_regret()
    for sc, _ in SC:
        for e in C.ESTIMATORS:
            g = fs[(fs.scenario == sc) & (fs.estimator == e)]
            a, b = g.loss_mix - g.best0, g.loss_mix - g.best1
            static = format_estimate(a.mean(), standard_error(a, scale=2), "7.2f")
            switch = format_estimate(b.mean(), standard_error(b, scale=2), "6.2f")
            verbose_print(f"{sc:9s} {e:8s} static {static}   switch1 {switch}")


def main():
    os.makedirs(ce.OUT, exist_ok=True)
    for e in C.ESTIMATORS:
        plot_scoresfam_heatmap(e)
    plot_density_bars()
    plot_regret_summary()
    plot_regret_jump()
    print_regret()


if __name__ == "__main__":
    main()
