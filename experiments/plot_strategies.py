"""Candidate figures for Section 6.3.2 (fixed vs VFF vs MFF), from the KLL-margin strategy runs (probe_strategies.cpp;
60 seeds, forecasts held before each observation):
  time series (Gaussian copula; results/strategies_paths.csv, PATHS=1 run, checked against strategies_jumppath / _seeds):
    ch6_strategies_rmse_time.png          pointwise RMSE = sqrt(mean over seeds of squared error), MSE smoothed by a
                                          centred 25-observation mean within each regime (never across the jump)
    ch6_strategies_excess_time.png        pointwise mean excess log loss, 101-observation smoothing within regimes;
                                          linear rows for constant / drift, symmetric-log jump row (linear below 10, log
                                          above) so the Hermite overshoot outliers stay on the axis: nothing clipped
    ch6_strategies_excess_time_median.png pointwise MEDIAN over seeds (a different quantity: typical, not expected, loss)
    ch6_strategies_h2_time.png            mean squared Hellinger at the checkpoints (every 50; every 10 just after the jump)
  bars (whole run, mean over the four families; results/strategies_summary.csv, ±2 SE):
    ch6_strategies_rmse_bars.png (3 x 3), ch6_strategies_rmse_bars_1x3.png
Fixed = hindsight-best whole-run fixed N_eff for that family, estimator, scenario and metric (strategies_summary).
Usage (repo root): python3 experiments/plot_strategies.py -> images/ch6/"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402  (RC of the approved Chapter 6 figures)

plt, core = ce.plt, ce.core
OUT = "images/ch6/"
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
# VFF = the development-selected rate alpha = 0.02 (strategy "vff_slow"); alpha = 0.1 is not shown.
# Colour = estimator (the Chapter 6 estimator colours); strategy = shade + line style: Fixed dark solid, VFF light
# dashed, MFF the full colour, thicker. The legend shows the three styles in neutral grey.
from matplotlib.colors import to_rgb  # noqa: E402
ECOL = {"rmpl": ce.COL["mle"], "xiao": ce.COL["xiao"], "hermite": ce.COL["hermite"]}
SHADE = {"fixed": ("dark", 0.45), "vff_slow": ("light", 0.5), "mff": ("full", 0.0)}


def shade(e, k):
    c, (kind, a) = np.array(to_rgb(ECOL[e])), SHADE[k]
    return tuple(c * (1 - a)) if kind == "dark" else tuple(c + (1 - c) * a) if kind == "light" else tuple(c)


STR = [("fixed", "Fixed (hindsight)", "#3a3a3a", "-", 1.1), ("vff_slow", "VFF", "#b0b0b0", (0, (4, 2)), 1.2),
       ("mff", "MFF", "#7a7a7a", "-", 1.6)]
JUMP, J0, J1 = 2500, 2200, 3000
summ = pd.read_csv(core.R + "strategies_summary.csv")
paths = pd.read_csv(core.R + "strategies_paths.csv")
os.makedirs(OUT, exist_ok=True)


def best_fixed(metric, sc, e, fam="gaussian"):
    r = summ[(summ.metric == metric) & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
             & (summ.copula == fam) & (summ.strategy == "fixed")]
    return r.best_fixed.iloc[0]


def smooth(t, y, w, sc):
    """Centred rolling mean, separately before and after the jump (whole series for constant / drift)."""
    s = pd.Series(np.asarray(y, float), index=t)
    parts = [s[s.index < JUMP], s[s.index >= JUMP]] if sc == "jump" else [s]
    return pd.concat([p.rolling(w, center=True, min_periods=1).mean() for p in parts]).values


def grid(fig_h=5.4):
    fig, axs = plt.subplots(3, 3, figsize=(core.W, fig_h), sharex="row", sharey="row")
    return fig, axs


def finish(fig, axs, ylab, out, legend_extra=None):
    for (sc, scl), row in zip(SC, axs):
        row[0].set_ylabel(f"{scl}\n{ylab}")
        for ax in row:
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            if sc == "jump":
                ax.axvline(JUMP, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=5)
                ax.set_xlim(J0, J1); ax.set_xticks([2250, 2500, 2750])
            else:
                ax.set_xlim(500, 5000); ax.set_xticks([1000, 2500, 4000])
    for ax, (_, el) in zip(axs[0], EST):
        ax.set_title(el)
    h = [plt.Line2D([], [], color=col, ls=ls, lw=lw) for _, _, col, ls, lw in STR]
    l = [lab for _, lab, _, _, _ in STR]
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=2.2)
    fig.text(0.53, 0.01, "Observation number, $n$", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.08, hspace=0.42, wspace=0.1)
    fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def series(sc, e, strategy, metric):
    key = best_fixed(metric, sc, e) if strategy == "fixed" else strategy
    g = paths[(paths.scenario == sc) & (paths.estimator == e) & (paths.strategy == key)].sort_values("t")
    if sc == "jump": g = g[(g.t >= J0) & (g.t <= J1)]
    return g


def rmse_time():
    with plt.rc_context(ce.RC):
        fig, axs = grid()
        for (sc, _), row in zip(SC, axs):
            for ax, (e, _) in zip(row, EST):
                for k, lab, col, ls, lw in STR:
                    g = series(sc, e, k, "rmse")
                    ax.plot(g.t, 100 * np.sqrt(smooth(g.t.values, g.mse, 25, sc)), color=shade(e, k), ls=ls, lw=lw, zorder=3)
            row[0].set_ylim(0, {"constant": 6, "drift": 9, "jump": 62}[sc])
        finish(fig, axs, r"RMSE ($\times10^{-2}$)", "ch6_strategies_rmse_time.png")


def excess_time(median=False):
    col_ = "ex_median" if median else "ex_mean"
    with plt.rc_context(ce.RC):
        fig, axs = grid()
        for (sc, _), row in zip(SC, axs):
            for ax, (e, _) in zip(row, EST):
                for k, lab, col, ls, lw in STR:
                    g = series(sc, e, k, "excess_x1000")
                    ax.plot(g.t, smooth(g.t.values, g[col_], 101, sc), color=shade(e, k), ls=ls, lw=lw, zorder=3)
                if not median and sc == "jump":   # the Hermite overshoot outliers live here: nothing is clipped
                    ax.set_yscale("symlog", linthresh=10, linscale=1.0)
            if median:
                row[0].set_ylim({"constant": (-1, 5), "drift": (-2, 30), "jump": (-5, 500)}[sc])
                if sc == "jump": row[0].set_yscale("symlog", linthresh=10)
            else:
                row[0].set_ylim({"constant": (-1, 6), "drift": (-1, 13), "jump": (-5, 30000)}[sc])
        lab = "Median excess\n" if median else "Excess log loss\n"
        finish(fig, axs, r"($\times10^{-3}$ nats)", "ch6_strategies_excess_time_median.png" if median else "ch6_strategies_excess_time.png")


def h2_time():
    with plt.rc_context(ce.RC):
        fig, axs = grid()
        for (sc, _), row in zip(SC, axs):
            top = 0.0
            for ax, (e, _) in zip(row, EST):
                for k, lab, col, ls, lw in STR:
                    g = series(sc, e, k, "h2_x1000").dropna(subset=["h2_mean"])
                    segs = [g[g.t < JUMP], g[g.t >= JUMP]] if sc == "jump" else [g]   # no line across the jump
                    for j, gg in enumerate(segs):
                        ax.plot(gg.t, gg.h2_mean, color=shade(e, k), ls=ls, lw=lw, zorder=3)
                    top = max(top, g.h2_mean.max())
            row[0].set_ylim(0, {"constant": 1.6, "drift": 2.5}.get(sc, 1.05 * top))
        finish(fig, axs, r"$H^2$ ($\times10^{-3}$)", "ch6_strategies_h2_time.png")


def bar_legend():
    """Strategy shades shown on a neutral grey: dark = Fixed, light = VFF, mid = MFF (each panel in its estimator's colour)."""
    g = np.array(to_rgb("#8a8a8a"))
    cols = [tuple(g * 0.55), tuple(g + (1 - g) * 0.5), tuple(g)]
    from matplotlib.patches import Patch
    return [Patch(color=c) for c in cols], [lab for _, lab, _, _, _ in STR]


def bars_3x3():
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(core.W, 5.0), sharey="row")
        for (sc, scl), row in zip(SC, axs):
            for ax, (e, el) in zip(row, EST):
                for i, (k, lab, col, ls, lw) in enumerate(STR):
                    r = summ[(summ.metric == "rmse") & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
                             & (summ.copula == "mean4") & (summ.strategy == k)].iloc[0]
                    ax.bar(i, r["mean"], width=0.75, color=shade(e, k), edgecolor="white", lw=0.5, zorder=3)
                    ax.errorbar(i, r["mean"], yerr=2 * r["se"], color="#52514e", lw=0.8, capsize=0, zorder=4)
                ax.set_xticks([]); ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            row[0].set_ylabel(f"{scl}\n" + r"RMSE ($\times10^{-2}$)")
            row[0].set_ylim(0, {"constant": 5.5, "drift": 7, "jump": 11}[sc])
        for ax, (_, el) in zip(axs[0], EST): ax.set_title(el)
        fig.legend(*bar_legend(), loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
        fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.03, hspace=0.3, wspace=0.1)
        fig.savefig(OUT + "ch6_strategies_rmse_bars.png"); plt.close(fig)
    print("wrote", OUT + "ch6_strategies_rmse_bars.png")


def bars_1x3():
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.5))
        bw = 0.26
        for ax, (sc, scl) in zip(axs, SC):
            for ei, (e, el) in enumerate(EST):
                for i, (k, lab, col, ls, lw) in enumerate(STR):
                    r = summ[(summ.metric == "rmse") & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
                             & (summ.copula == "mean4") & (summ.strategy == k)].iloc[0]
                    x = ei + (i - 1) * bw
                    ax.bar(x, r["mean"], width=bw * 0.92, color=shade(e, k), edgecolor="white", lw=0.5, zorder=3)
                    ax.errorbar(x, r["mean"], yerr=2 * r["se"], color="#52514e", lw=0.7, capsize=0, zorder=4)
            ax.set_xticks(range(3)); ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"], fontsize=7)
            ax.tick_params(axis="x", length=0); ax.set_title(scl)
            ax.set_ylim(0, {"constant": 5.5, "drift": 7, "jump": 11}[sc])
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
        fig.legend(*bar_legend(), loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
        fig.subplots_adjust(left=0.08, right=0.985, top=0.8, bottom=0.17, wspace=0.25)
        fig.savefig(OUT + "ch6_strategies_rmse_bars_1x3.png"); plt.close(fig)
    print("wrote", OUT + "ch6_strategies_rmse_bars_1x3.png")


if __name__ == "__main__":
    for sc, _ in SC:
        print(sc, {e: {m: best_fixed(m, sc, e) for m in ["rmse", "excess_x1000", "h2_x1000"]} for e, _ in EST})
    rmse_time(); excess_time(); excess_time(median=True); h2_time(); bars_3x3(); bars_1x3()
