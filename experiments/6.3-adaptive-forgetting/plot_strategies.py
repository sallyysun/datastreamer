"""Section 6.3 trajectory figures (fixed vs VFF vs MFF), Gaussian copula, from results/strategies_paths.csv (PATHS=1),
results/strategies_paths_est.csv (PATHS=1 TAG=est) and results/strategies_summary.csv (analyze_strategies.py). Pointwise
RMSE = sqrt(mean over seeds of squared error), MSE smoothed within regimes (never across the jump). Fixed = hindsight-best
whole-run fixed N_eff; VFF = alpha 0.02.
Usage: python3 plot_strategies.py -> images/ch6_rmse_trajectories.png, images/ch6_jump_trajectories.png"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
OUT = "images/"
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
# VFF = the development-selected rate alpha = 0.02 (strategy "vff_slow"); alpha = 0.1 is not shown.
# Colour = strategy (Fixed charcoal solid, VFF purple dashed, MFF crimson solid), the same in every panel.


# One colour per strategy, the same in every panel (outside the green / blue / orange estimator palette):
STRCOL = C.STRATEGY_COLOUR


def shade(e, k):
    return STRCOL[k]


STR = [("fixed", "Fixed (hindsight)", STRCOL["fixed"], "-", 1.1), ("vff_slow", "VFF", STRCOL["vff_slow"], (0, (4, 2)), 1.2),
       ("mff", "MFF", STRCOL["mff"], "-", 1.1)]
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


if __name__ == "__main__":
    rmse_time("ch6_rmse_trajectories.png")
    jump_trajectories("ch6_jump_trajectories.png")
