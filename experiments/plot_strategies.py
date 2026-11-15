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
# Colour = strategy (Fixed charcoal solid, VFF purple dashed, MFF crimson solid), the same in every panel.


# One colour per strategy, the same in every panel (outside the green / blue / orange estimator palette):
STRCOL = {"fixed": "#333333", "vff_slow": "#8E6BB8", "mff": "#C2185B"}


def shade(e, k):
    return STRCOL[k]


STR = [("fixed", "Fixed (hindsight)", STRCOL["fixed"], "-", 1.1), ("vff_slow", "VFF", STRCOL["vff_slow"], (0, (4, 2)), 1.2),
       ("mff", "MFF", STRCOL["mff"], "-", 1.1)]
if os.environ.get("VFF_SOLID"):   # VFF as a solid purple line (colour alone separates the strategies)
    STR = [(k, lab, col, "-" if k == "vff_slow" else ls, lw) for k, lab, col, ls, lw in STR]
JUMP, J0, J1 = 2500, 2200, 3000
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""   # mix: MFF density = mixture (MIXDENS runs)
summ = pd.read_csv(core.R + f"strategies_summary{T}.csv")
paths = pd.read_csv(core.R + f"strategies_paths{T}.csv")
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


def rmse_time(out="ch6_strategies_rmse_time.png"):
    with plt.rc_context(ce.RC):
        fig, axs = grid()
        for (sc, _), row in zip(SC, axs):
            for ax, (e, _) in zip(row, EST):
                for k, lab, col, ls, lw in STR:
                    g = series(sc, e, k, "rmse")
                    ax.plot(g.t, 100 * np.sqrt(smooth(g.t.values, g.mse, 25, sc)), color=shade(e, k), ls=ls, lw=lw, zorder=3)
            row[0].set_ylim(0, {"constant": 6, "drift": 9, "jump": 62}[sc])
        finish(fig, axs, r"RMSE ($\times10^{-2}$)", out)


def excess_time(median=False, out=None):
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
        out = out or ("ch6_strategies_excess_time_median.png" if median else "ch6_strategies_excess_time.png")
        finish(fig, axs, (r"Median excess ($\times10^{-3}$)" if median else r"Excess log loss ($\times10^{-3}$)"), out)


def h2_time(out="ch6_strategies_h2_time.png"):
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
        finish(fig, axs, r"$H^2$ ($\times10^{-3}$)", out)


def bar_legend():
    """Strategy shades shown on a neutral grey: dark = Fixed, light = VFF, mid = MFF (each panel in its estimator's colour)."""
    from matplotlib.patches import Patch
    return [Patch(color=col) for _, _, col, _, _ in STR], [lab for _, lab, _, _, _ in STR]


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


def bars_1x3(out="ch6_strategies_rmse_bars_1x3.png"):
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
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def jump_adaptation(out="ch6_jump_adaptation.png"):
    """Main-text Figure 2: abrupt jump only, Gaussian copula (as the other time-series figures). Columns = estimators.
    Row 1: mean over the 60 seeds of the prequential forecast of rho_S, with the true path (shared by all families).
    Row 2: pointwise RMSE across the 60 seeds (sqrt of the mean squared error at each n, not the error of the mean path),
    MSE smoothed over a centred 25-observation window within each regime, never across the jump.
    Fixed = the Gaussian whole-run RMSE-best N_eff (199 for every estimator), held constant."""
    pe = pd.read_csv(core.R + "strategies_paths_est.csv")
    pe = pe[(pe.scenario == "jump") & (pe.t >= J0) & (pe.t <= J1)]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(core.W, 3.9), sharex=True, sharey="row")
        for j, (e, el) in enumerate(EST):
            for k, lab, col, ls, lw in STR:
                key = best_fixed("rmse", "jump", e) if k == "fixed" else k
                g = pe[(pe.estimator == e) & (pe.strategy == key)].sort_values("t")
                rm = 100 * np.sqrt(smooth(g.t.values, g.mse, 25, "jump"))
                for part in (g.t < JUMP, g.t >= JUMP):
                    axs[0, j].plot(g.t[part], g.est_mean[part], color=col, ls=ls, lw=lw, zorder=4)
                    axs[1, j].plot(g.t[part], rm[part.values], color=col, ls=ls, lw=lw, zorder=4)
            for part in (g.t < JUMP, g.t >= JUMP):
                axs[0, j].plot(g.t[part], g.rho_true[part], color="#9a9a9a", lw=2.4, alpha=0.6, zorder=3, solid_capstyle="butt")
            axs[0, j].set_title(el)
            for ax in axs[:, j]:
                ax.axvline(JUMP, color="#8a8a8a", lw=0.7, ls=(0, (4, 3)), zorder=2)
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                ax.set_xlim(J0, J1); ax.set_xticks([2250, 2500, 2750])
        axs[0, 0].set_ylim(0, 0.9); axs[1, 0].set_ylim(0, 62)
        axs[0, 0].set_ylabel(r"Mean $\hat\rho_S$"); axs[1, 0].set_ylabel(r"RMSE ($\times10^{-2}$)")
        h = [plt.Line2D([], [], color="#9a9a9a", lw=2.4, alpha=0.6)] + [plt.Line2D([], [], color=c, ls=st, lw=w) for _, _, c, st, w in STR]
        fig.legend(h, [r"True $\rho_S$"] + [lab for _, lab, _, _, _ in STR], loc="upper center", ncol=4, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=2.2)
        fig.text(0.53, 0.01, "Observation number, $n$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.12, hspace=0.25, wspace=0.1)
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def density_bars(fam="gaussian", out="ch6_gaussian_density_bars.png"):
    """Main-text Figure 3: whole-run excess log loss (row 1) and squared Hellinger (row 2) by scenario (columns), estimator
    (groups) and strategy (bars), Gaussian copula. Density = the Gaussian copula at each forecast rho_S (param_from_spearman:
    r = 2 sin(pi rho / 6)). Fixed = hindsight-best fixed N_eff chosen separately for each metric (strategies_summary
    best_fixed for that family, estimator and scenario). Bars: mean over 60 seeds; whiskers ± 2 SE. Nothing clipped."""
    MET = [("excess_x1000", r"Excess log loss ($\times10^{-3}$)"), ("h2_x1000", r"$H^2$ ($\times10^{-3}$)")]
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(core.W, 4.2))
        for ri, (m, ylab) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]
                top = 0.0
                for ei, (e, el) in enumerate(EST):
                    for i, (k, lab, col, ls, lw) in enumerate(STR):
                        r = summ[(summ.metric == m) & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
                                 & (summ.copula == fam) & (summ.strategy == k)].iloc[0]
                        x = ei + (i - 1) * bw
                        ax.bar(x, r["mean"], width=bw * 0.92, color=col, edgecolor="white", lw=0.5, zorder=3)
                        ax.errorbar(x, r["mean"], yerr=2 * r["se"], color="#52514e", lw=0.7, capsize=0, zorder=4)
                        top = max(top, r["mean"] + 2 * r["se"])
                if m == "excess_x1000": ax.axhline(0, color="#333333", lw=0.7, zorder=2)
                ax.set_ylim(min(0, ax.get_ylim()[0]), 1.08 * top)
                ax.set_xticks(range(3)); ax.tick_params(axis="x", length=0)
                ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"] if ri == 1 else [], fontsize=7)
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if ri == 0: ax.set_title(scl)
            axs[ri, 0].set_ylabel(ylab)
        fig.legend(*bar_legend(), loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
        fig.subplots_adjust(left=0.1, right=0.985, top=0.87, bottom=0.12, hspace=0.25, wspace=0.28)
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def reorganised():
    """The Chapter 6.3.2 figure set (2026-11-12 reorganisation)."""
    bars_1x3("ch6_strategy_rmse_bars.png")                       # main Figure 1 (four-family mean)
    jump_adaptation("ch6_jump_adaptation.png")                   # main Figure 2 (Gaussian)
    density_bars("gaussian", "ch6_gaussian_density_bars.png")    # main Figure 3 (Gaussian)
    est_time(band=False, out="ch6_appendix_est_trajectories.png")  # appendix A (Gaussian, mean lines only)
    rmse_time("ch6_appendix_rmse_trajectories.png")               # appendix B (Gaussian)


if __name__ == "__main__":
    for sc, _ in SC:
        print(sc, {e: {m: best_fixed(m, sc, e) for m in ["rmse", "excess_x1000", "h2_x1000"]} for e, _ in EST})
    rmse_time(); excess_time(); excess_time(median=True); h2_time(); bars_3x3(); bars_1x3()


def est_time(band=True, out="ch6_strategies_est_time.png"):
    """Figure A variant: mean prequential estimate of rho_S over the 60 seeds (and the 10-90% spread across seeds) against
    the true rho_S path. Gaussian copula; results/strategies_paths_est.csv (PATHS=1 TAG=est rerun, identical per-tick
    MSE). Fixed = Gaussian whole-run RMSE-best N_eff. No smoothing."""
    pe = pd.read_csv(core.R + "strategies_paths_est.csv")
    with plt.rc_context(ce.RC):
        fig, axs = grid()
        for (sc, _), row in zip(SC, axs):
            for ax, (e, _) in zip(row, EST):
                tr = None
                for k, lab, col, ls, lw in STR:
                    key = best_fixed("rmse", sc, e) if k == "fixed" else k
                    g = pe[(pe.scenario == sc) & (pe.estimator == e) & (pe.strategy == key)].sort_values("t")
                    if sc == "jump": g = g[(g.t >= J0) & (g.t <= J1)]
                    tr = g
                    segs = [g[g.t < JUMP], g[g.t >= JUMP]] if sc == "jump" else [g]
                    for gg in segs:
                        if band: ax.fill_between(gg.t, gg.est_q10, gg.est_q90, color=col, alpha=0.10, lw=0, zorder=2)
                        ax.plot(gg.t, gg.est_mean, color=col, ls=ls, lw=lw, zorder=4)
                for gg in ([tr[tr.t < JUMP], tr[tr.t >= JUMP]] if sc == "jump" else [tr]):
                    ax.plot(gg.t, gg.rho_true, color="#9a9a9a", lw=2.4, alpha=0.6, zorder=3, solid_capstyle="butt")
            row[0].set_ylim({"constant": (0.45, 0.7), "drift": (0.0, 0.9), "jump": (0.0, 0.9)}[sc])
        extra = ([plt.Line2D([], [], color="#9a9a9a", lw=2.4, alpha=0.6)], [r"True $\rho_S$"])
        h = [plt.Line2D([], [], color="#9a9a9a", lw=2.4, alpha=0.6)] + [plt.Line2D([], [], color=c, ls=s, lw=w) for _, _, c, s, w in STR]
        l = [r"True $\rho_S$"] + [lab for _, lab, _, _, _ in STR]
        for (sc, scl), r in zip(SC, axs):
            r[0].set_ylabel(f"{scl}\n" + r"$\hat\rho_S$")
            for ax in r:
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if sc == "jump":
                    ax.axvline(JUMP, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=5)
                    ax.set_xlim(J0, J1); ax.set_xticks([2250, 2500, 2750])
                else:
                    ax.set_xlim(500, 5000); ax.set_xticks([1000, 2500, 4000])
        for ax, (_, el) in zip(axs[0], EST): ax.set_title(el)
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=2.2)
        fig.text(0.53, 0.01, "Observation number, $n$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.08, hspace=0.42, wspace=0.1)
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def accuracy_bars(fam="gaussian", out="ch6_accuracy_bars.png"):
    """6.3.2 Figure 1: 3 x 3 grouped bars, rows = Spearman RMSE / excess log loss / squared Hellinger, columns =
    scenarios, groups = estimators, bars = Fixed / VFF (alpha = 0.02) / MFF; Gaussian copula; whole run; ± 2 SE.
    Fixed = hindsight-best fixed N_eff chosen separately for each metric (strategies_summary best_fixed for that family,
    estimator and scenario). Density forecasts: the Gaussian copula at each forecast rho_S. Every panel has its own y scale
    (levels differ by up to an order of magnitude between scenarios), starting at 0. Nothing clipped."""
    MET = [("rmse", r"Spearman RMSE ($\times10^{-2}$)", 100), ("excess_x1000", r"Excess log loss ($\times10^{-3}$)", 1),
           ("h2_x1000", r"$H^2$ ($\times10^{-3}$)", 1)]
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(core.W, 5.6))
        for ri, (m, ylab, sc_) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]; top = 0.0
                for ei, (e, el) in enumerate(EST):
                    for i, (k, lab, col, ls, lw) in enumerate(STR):
                        r = summ[(summ.metric == m) & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
                                 & (summ.copula == fam) & (summ.strategy == k)].iloc[0]
                        x = ei + (i - 1) * bw
                        ax.bar(x, r["mean"], width=bw * 0.92, color=col, edgecolor="white", lw=0.5, zorder=3)
                        ax.errorbar(x, r["mean"], yerr=2 * r["se"], color="#52514e", lw=0.7, capsize=0, zorder=4)
                        top = max(top, r["mean"] + 2 * r["se"])
                if m == "excess_x1000": ax.axhline(0, color="#333333", lw=0.7, zorder=2)
                ax.set_ylim(0, 1.08 * top)
                ax.set_xticks(range(3)); ax.tick_params(axis="x", length=0)
                ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"] if ri == 2 else [], fontsize=7)
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if ri == 0: ax.set_title(scl)
            axs[ri, 0].set_ylabel(ylab)
        fig.legend(*bar_legend(), loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
        fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.09, hspace=0.22, wspace=0.28)
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def jump_trajectories(out="ch6_jump_trajectories.png"):
    """6.3.2 Figure 3: 1 x 3, abrupt jump, Gaussian copula: mean over 60 seeds of the prequential forecast of rho_S for
    Fixed (Gaussian RMSE-best, N_eff 199), VFF and MFF, with the true path; dashed line at the jump; shared y axis."""
    pe = pd.read_csv(core.R + "strategies_paths_est.csv")
    pe = pe[(pe.scenario == "jump") & (pe.t >= J0) & (pe.t <= J1)]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.5), sharey=True)
        for ax, (e, el) in zip(axs, EST):
            for k, lab, col, ls, lw in STR:
                key = best_fixed("rmse", "jump", e) if k == "fixed" else k
                g = pe[(pe.estimator == e) & (pe.strategy == key)].sort_values("t")
                for part in (g.t < JUMP, g.t >= JUMP):
                    ax.plot(g.t[part], g.est_mean[part], color=col, ls=ls, lw=lw, zorder=4)
            for part in (g.t < JUMP, g.t >= JUMP):
                ax.plot(g.t[part], g.rho_true[part], color="#9a9a9a", lw=2.4, alpha=0.6, zorder=3, solid_capstyle="butt")
            ax.axvline(JUMP, color="#8a8a8a", lw=0.7, ls=(0, (4, 3)), zorder=2)
            ax.set_title(el); ax.set_xlim(J0, J1); ax.set_xticks([2250, 2500, 2750]); ax.set_ylim(0, 0.9)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Mean $\hat\rho_S$")
        h = [plt.Line2D([], [], color="#9a9a9a", lw=2.4, alpha=0.6)] + [plt.Line2D([], [], color=c, ls=st, lw=w) for _, _, c, st, w in STR]
        fig.legend(h, [r"True $\rho_S$"] + [lab for _, lab, _, _, _ in STR], loc="upper center", ncol=4, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=2.2)
        fig.text(0.53, 0.02, "Observation number, $n$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.8, bottom=0.2, wspace=0.1)
        fig.savefig(OUT + out); plt.close(fig)
    print("wrote", OUT + out)


def section_63():
    """Section 6.3 figure set (2026-11-12 brief). Figures 4-5 (weighting loss, weights / fixed-share) need KLL runs that
    don't exist yet."""
    accuracy_bars("gaussian", "ch6_accuracy_bars.png")                    # Fig. 1 (Gaussian, 3 metrics)
    rmse_time("ch6_rmse_trajectories.png")                                # Fig. 2 (Gaussian, main text)
    jump_trajectories("ch6_jump_trajectories.png")                        # Fig. 3 (Gaussian, jump)
    excess_time(out="ch6_excess_trajectories.png")                        # as Fig. 2, excess log loss (Fixed by excess)
    h2_time("ch6_h2_trajectories.png")                                    # as Fig. 2, squared Hellinger (Fixed by H^2)
    bars_1x3("ch6_appendix_rmse_bars_4fam.png")                           # appendix: four-family RMSE bars
    est_time(band=True, out="ch6_appendix_est_trajectories_band.png")     # appendix: full 3x3 with 10-90% bands


def density_63():
    """The Section 6.3 figures whose MFF numbers depend on the density forecast (excess log loss, H^2), written with the
    DENS_TAG suffix: DENS_TAG=mix python3 -c 'import plot_strategies as p; p.density_63()' -> ch6_*_mix.png. The RMSE
    rows / figures are identical to the untagged ones."""
    accuracy_bars("gaussian", f"ch6_accuracy_bars{T}.png")
    density_bars("gaussian", f"ch6_gaussian_density_bars{T}.png")
    excess_time(out=f"ch6_excess_trajectories{T}.png")
    h2_time(f"ch6_h2_trajectories{T}.png")


def solid_63():
    """The 6.3 trajectory figures with VFF solid: VFF_SOLID=1 DENS_TAG=mix python3 -c 'import plot_strategies as p; p.solid_63()'
    -> *_solid.png (excess / H^2 from the mixture-density run; RMSE / estimates identical either way)."""
    rmse_time("ch6_rmse_trajectories_solid.png")
    jump_trajectories("ch6_jump_trajectories_solid.png")
    excess_time(out=f"ch6_excess_trajectories{T}_solid.png")
    h2_time(f"ch6_h2_trajectories{T}_solid.png")
    est_time(band=False, out="ch6_appendix_est_trajectories_solid.png")
