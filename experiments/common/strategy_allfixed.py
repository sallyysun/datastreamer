"""Plot all fixed and adaptive strategies for one experiment section."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [(e, C.EST_LABEL2[e]) for e in C.ESTIMATORS]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
FIX = [str(m) for m in C.MEMORIES]
BARS = [f"fixed_{n}" for n in FIX] + ["vff_slow", "mff"]
C_OTHER, C_BEST, C_VFF, C_MFF = "#cfcfcf", "#333333", "#8E6BB8", "#C2185B"
CFG = {"rmse": ("strategies_seeds.csv", "rmse", 100, r"Spearman RMSE ($\times10^{-2}$)", {"constant": 8, "drift": 11, "jump": 14}),
       "h2": ("strategies_seeds_mix.csv", "h2_x1000", 1, r"Squared Hellinger ($\times10^{-3}$)", {"constant": 3.5, "drift": 4.5, "jump": 8}),
       "excess": ("strategies_seeds_mix.csv", "excess_x1000", 1, r"Excess log loss ($\times10^{-3}$)", {"constant": 6, "drift": 8, "jump": 20})}


def load(section_dir, metric):
    src, col, scale, _, _ = CFG[metric]
    d = pd.read_csv(section_dir / "results" / src)
    d = d[(d.copula == "gaussian") & (d.window == "whole")]
    a = d.groupby(["scenario", "estimator", "strategy"])[col].agg(["mean", "std", "count"])
    a["se"] = a["std"] / np.sqrt(a["count"])
    a[["mean", "se"]] *= scale
    return a


def panel(ax, agg, sc, ym, metric):
    bw = 0.115
    for ei, (e, _) in enumerate(EST):
        a = agg.loc[(sc, e)]
        best = a.loc[[f"fixed_{n}" for n in FIX], "mean"].idxmin()
        prev = -1   # stagger level of the previous cut bar's label, so adjacent labels do not overlap
        for i, k in enumerate(BARS):
            m, se = a.loc[k, "mean"], a.loc[k, "se"]
            x = ei + (i - 3) * bw
            col = C_VFF if k == "vff_slow" else C_MFF if k == "mff" else C_BEST if k == best else C_OTHER
            ax.bar(x, min(m, ym), width=bw * 0.9, color=col, edgecolor="white", lw=0.4, zorder=3)
            if m > ym:
                ax.plot([x - bw * 0.45, x + bw * 0.45], [ym * 0.965, ym * 0.985], color="white", lw=1.2, zorder=5)
                lab = f"{m:.0f}" if m >= 9.95 or metric == "rmse" else f"{m:.1f}"
                lev = 1 if prev == 0 else 0
                ax.text(x, ym * (0.99 + 0.075 * lev), lab, ha="center", va="bottom", fontsize=5.2, color="#52514e", clip_on=False)
                prev = lev
            else:
                prev = -1
                ax.errorbar(x, m, yerr=2 * se, color="#52514e", lw=0.6, capsize=0, zorder=4)
            if k == best:
                ax.text(x, min(m + 2 * se, ym) + 0.012 * ym, k.split("_")[1], ha="center", va="bottom",
                        fontsize=5.2, color=C_BEST, rotation=90)
    ax.set_ylim(0, ym)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3))
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def legend(fig, y):
    def sq(c):
        return plt.Rectangle((0, 0), 1, 1, color=c)
    fig.legend([sq(C_OTHER), sq(C_BEST), sq(C_VFF), sq(C_MFF)],
               [r"Fixed $N_{\mathrm{eff}}$ = " + ", ".join(FIX) + " (left to right)", "Fixed, hindsight best (labelled)",
                "VFF", "MFF"], loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.53, y),
               handlelength=1.2, columnspacing=1.5)


def density_fig(section_dir):
    """2 x 3: squared Hellinger (top) and excess log loss (bottom)."""
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(ce.W, 4.9))
        for ri, met in enumerate(["h2", "excess"]):
            agg = load(section_dir, met)
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]
                panel(ax, agg, sc, CFG[met][4][sc], met)
                ax.set_xticklabels([el for _, el in EST] if ri == 1 else [], fontsize=7)
                if ri == 0:
                    ax.set_title(scl, pad=15)
            axs[ri, 0].set_ylabel(CFG[met][3])
        legend(fig, 1.005)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.855, bottom=0.085, hspace=0.26, wspace=0.22)
        out = section_dir / "images" / "ch6_strategy_density_bars_allfixed.png"
        fig.savefig(out)
        plt.close(fig)


def metric_fig(section_dir, metric):
    _, _, _, ylab, ymax = CFG[metric]
    agg = load(section_dir, metric)
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.7))
        for ax, (sc, scl) in zip(axs, SC):
            panel(ax, agg, sc, ymax[sc], metric)
            ax.set_xticklabels([el for _, el in EST], fontsize=7)
            ax.set_title(scl, pad=9 if metric == "rmse" else 15)
        axs[0].set_ylabel(ylab)
        legend(fig, 1.01)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.76 if metric == "rmse" else 0.73, bottom=0.15, wspace=0.22)
        out = section_dir / "images" / f"ch6_strategy_{metric}_bars_allfixed.png"
        fig.savefig(out)
        plt.close(fig)


def plot_all_fixed(section_dir, metric):
    if metric == "density":
        density_fig(section_dir)
    else:
        metric_fig(section_dir, metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section_dir", type=Path)
    parser.add_argument("metric", choices=("rmse", "density", "h2", "excess"))
    args = parser.parse_args()
    plot_all_fixed(args.section_dir, args.metric)


if __name__ == "__main__":
    main()
