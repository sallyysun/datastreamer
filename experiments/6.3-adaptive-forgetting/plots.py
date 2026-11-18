"""Section 6.3 figures: strategy RMSE bars and paths, the jump trajectories, MFF weight paths, and VFF rate sensitivity."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

sys.path.insert(0, "../common")
from strategy_allfixed import plot_all_fixed  # noqa: E402
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
JUMP, J0, J1 = C.CHANGE, 2200, 3000
STR = [("fixed", "Fixed (hindsight)", ce.STRATEGY["fixed"], "-", 1.1), ("vff_slow", "VFF", ce.STRATEGY["vff"], (0, (4, 2)), 1.2),
       ("mff", "MFF", ce.STRATEGY["mff"], "-", 1.1)]
MEM = list(ce.memory_colours(C.MEMORIES).items())
WC = [f"w{m}" for m, _ in MEM]
NEFF = r"$N_{\mathrm{eff}}$"
TIME_LABEL = "Observation number, $n$"
RMSE_LABEL = r"RMSE ($\times10^{-2}$)"


def scenario_grid(fig, axs, ylab, legend_handles, legend_labels, xlabel, hspace, handlelength=None):
    """3 x 3 grid: scenarios down, estimators across."""
    for (_, scl), row in zip(SC, axs):
        row[0].set_ylabel(f"{scl}\n{ylab}")
    for ax, e in zip(axs[0], C.ESTIMATORS):
        ax.set_title(C.EST_LABEL[e])
    ce.top_legend(fig, legend_handles, legend_labels, ncol=len(legend_labels), handlelength=handlelength)
    ce.bottom_label(fig, xlabel)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.08, hspace=hspace, wspace=0.1)


def strategy_handles():
    return [plt.Line2D([], [], color=col, ls=ls, lw=lw) for _, _, col, ls, lw in STR], [lab for _, lab, _, _, _ in STR]


def best_fixed(summ, metric, sc, e, fam="gaussian"):
    r = summ[(summ.metric == metric) & (summ.scenario == sc) & (summ.window == "whole") & (summ.estimator == e)
             & (summ.copula == fam) & (summ.strategy == "fixed")]
    return r.best_fixed.iloc[0]


def strategy_key(summ, metric, sc, e, strategy):
    return best_fixed(summ, metric, sc, e) if strategy == "fixed" else strategy


def smooth(t, y, w, sc):
    """Centred rolling mean, separately before and after the jump."""
    s = pd.Series(np.asarray(y, float), index=t)
    parts = [s[s.index < JUMP], s[s.index >= JUMP]] if sc == "jump" else [s]
    return pd.concat([p.rolling(w, center=True, min_periods=1).mean() for p in parts]).values


def jump_window(ax):
    ax.set_xlim(J0, J1)
    ax.set_xticks([2250, 2500, 2750])


def path_axes(ax, sc):
    ce.grid(ax, "y")
    if sc == "jump":
        ax.axvline(JUMP, color=ce.MUTED, lw=0.8, ls=(0, (4, 3)), zorder=5)
        jump_window(ax)
    else:
        ax.set_xlim(500, 5000)
        ax.set_xticks([1000, 2500, 4000])


def load_summary():
    return pd.read_csv(ce.R + "strategies_summary.csv")


def plot_strategy_bars():
    plot_all_fixed(Path("."), "rmse")


def plot_rmse_trajectories():
    """Smoothed RMSE by tick for hindsight fixed, VFF and MFF."""
    summ, paths = load_summary(), pd.read_csv(ce.R + "strategies_paths.csv")
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(ce.W, 5.4), sharex="row", sharey="row")
        for (sc, _), row in zip(SC, axs):
            for ax, e in zip(row, C.ESTIMATORS):
                for k, _, col, ls, lw in STR:
                    g = paths[(paths.scenario == sc) & (paths.estimator == e)
                              & (paths.strategy == strategy_key(summ, "rmse", sc, e, k))].sort_values("t")
                    if sc == "jump":
                        g = g[(g.t >= J0) & (g.t <= J1)]
                    ax.plot(g.t, 100 * np.sqrt(smooth(g.t.values, g.mse, 25, sc)), color=col, ls=ls, lw=lw, zorder=3)
                path_axes(ax, sc)
            row[0].set_ylim(0, {"constant": 6, "drift": 9, "jump": 62}[sc])
        scenario_grid(fig, axs, RMSE_LABEL, *strategy_handles(), TIME_LABEL, hspace=0.42, handlelength=2.2)
        ce.save(fig, "ch6_rmse_trajectories.png")


def plot_jump_trajectories():
    """Seed-mean forecast of rho_S through the jump, with the true path."""
    summ = load_summary()
    pe = pd.read_csv(ce.R + "strategies_paths_est.csv")
    pe = pe[(pe.scenario == "jump") & (pe.t >= J0) & (pe.t <= J1)]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.5), sharey=True)
        for ax, e in zip(axs, C.ESTIMATORS):
            for k, _, col, ls, lw in STR:
                g = pe[(pe.estimator == e) & (pe.strategy == strategy_key(summ, "rmse", "jump", e, k))].sort_values("t")
                for part in (g.t < JUMP, g.t >= JUMP):
                    ax.plot(g.t[part], g.est_mean[part], color=col, ls=ls, lw=lw, zorder=4)
            for part in (g.t < JUMP, g.t >= JUMP):
                ax.plot(g.t[part], g.rho_true[part], color=ce.REF, lw=2.4, alpha=0.6, zorder=3, solid_capstyle="butt")
            ax.axvline(JUMP, color=ce.REF, lw=0.7, ls=(0, (4, 3)), zorder=2)
            ax.set_title(C.EST_LABEL[e])
            jump_window(ax)
            ax.set_ylim(0, 0.9)
            ce.grid(ax, "y")
        axs[0].set_ylabel(r"Mean $\hat\rho_S$")
        h, l = strategy_handles()
        ce.top_legend(fig, [plt.Line2D([], [], color=ce.REF, lw=2.4, alpha=0.6)] + h, [r"True $\rho_S$"] + l, ncol=4, handlelength=2.2)
        ce.bottom_label(fig, TIME_LABEL, y=0.02)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.8, bottom=0.2, wspace=0.1)
        ce.save(fig, "ch6_jump_trajectories.png")


def weight_path(W, sc, e, delta):
    g = W[(W.scenario == sc) & (W.estimator == e) & np.isclose(W.delta, delta)].sort_values("t").copy()
    g[WC] = g[WC].div(g[WC].sum(axis=1), axis=0)   # the seed-mean weights sum to 1 only up to float rounding
    g["nbar"] = g[WC].values @ np.array([m for m, _ in MEM], float)
    return g


def weight_panel(a0, a1, g, sc):
    a0.stackplot(g.t, *[g[k] for k in WC], colors=[col for _, col in MEM], lw=0, zorder=2)
    a0.set_ylim(0, 1)
    for p in ([g[g.t < JUMP], g[g.t >= JUMP]] if sc == "jump" else [g]):
        a1.plot(p.t, p.nbar, color=ce.DARK, lw=1.1, zorder=4)
    a1.set_yscale("log")
    a1.set_ylim(30, 2600)
    a1.set_yticks([m for m, _ in MEM])
    a1.set_yticklabels([str(m) for m, _ in MEM])
    a1.minorticks_off()
    ce.grid(a1, "y")
    for ax in (a0, a1):
        ax.set_xlim(C.BURN, C.T)
        ax.set_xticks([1000, JUMP, 4000])
        if sc == "jump":
            ax.axvline(JUMP, color=ce.DARK, lw=0.8, ls=(0, (4, 3)), zorder=6)


def weight_legend(fig, e, x):
    h = [ce.swatch(color=c) for _, c in MEM]
    ce.top_legend(fig, h, [r"$N_{\mathrm{eff}}=%d$" % m for m, _ in MEM], ncol=5, handlelength=1.2, title=f"{C.EST_LABEL[e]} estimator", title_fontsize=7, x=x)


def load_weights():
    W = pd.read_csv(ce.R + "strategies_weights.csv")
    return W[(W.t >= C.BURN) & (W.t < C.T)]


def plot_mff_weights(e, W):
    """Stacked MFF weights and the weight-averaged memory, per scenario."""
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(ce.W, 3.1), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
        for j, (sc, scl) in enumerate(SC):
            weight_panel(axs[0, j], axs[1, j], weight_path(W, sc, e, 1e-4), sc)
            axs[0, j].set_title(scl)
        axs[0, 0].set_ylabel("Mean MFF weight")
        axs[1, 0].set_ylabel("Weight-averaged\ncandidate " + NEFF)
        weight_legend(fig, e, 0.53)
        ce.bottom_label(fig, TIME_LABEL)
        fig.subplots_adjust(left=0.12, right=0.985, top=1 - 0.665 / 3.1, bottom=0.42 / 3.1, hspace=0.14, wspace=0.08)
        ce.save(fig, f"ch6_mff_weights_{e}.png")


def plot_fixed_share(e, W):
    """Columns plain Hedge | fixed share; for the jump, a stacked-weight row and a candidate-N_eff row."""
    scen = SC[2:]
    rows = 2 * len(scen)
    with plt.rc_context(ce.RC):
        H = 1.05 * rows + 0.85
        fig, axs = plt.subplots(rows, 2, figsize=(ce.W * 0.8, H), sharex=True, gridspec_kw={"height_ratios": [1.35, 1] * len(scen)})
        for c, (delta, cl) in enumerate([(0.0, r"Ordinary Hedge ($\delta=0$)"), (1e-4, r"Fixed share ($\delta=10^{-4}$)")]):
            for si, (sc, scl) in enumerate(scen):
                a0, a1 = axs[2 * si, c], axs[2 * si + 1, c]
                weight_panel(a0, a1, weight_path(W, sc, e, delta), sc)
                a0.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                if c == 0:
                    a0.set_ylabel(f"{scl}\nmean weight")
                    a1.set_ylabel("Weight-averaged\ncandidate " + NEFF)
                else:
                    a0.set_yticklabels([])
                    a1.set_yticklabels([])
            axs[0, c].set_title(cl)
        weight_legend(fig, e, 0.55)
        ce.bottom_label(fig, TIME_LABEL, x=0.56, y=0.008)
        fig.subplots_adjust(left=0.15, right=0.98, top=1 - 0.66 / H, bottom=0.45 / H, hspace=0.16, wspace=0.06)
        ce.save(fig, f"ch6_mff_fixedshare_jump_{e}.png")


def vffsens_panel(ax, g, e):
    c = np.array(to_rgb(ce.ESTIMATOR[e]))
    dark = tuple(c * 0.55)
    m, s = g.loc["mff", "mean"], g.loc["mff", "se"]
    ax.axhspan(m - 2 * s, m + 2 * s, color=dark, alpha=0.15, lw=0, zorder=1)
    ax.axhline(m, color=dark, lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.axvline(0.02, color=ce.REF, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    y = [g.loc[f"vff_{a:g}", "mean"] for a in C.VFF_RATES]
    e2 = [2 * g.loc[f"vff_{a:g}", "se"] for a in C.VFF_RATES]
    ax.errorbar(C.VFF_RATES, y, yerr=e2, color=tuple(c), lw=1.2, marker="o", ms=3, capsize=0, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.005, 0.02, 0.1])
    ax.set_xticklabels(["0.005", "0.02", "0.1"])
    ax.minorticks_off()
    ce.grid(ax, "y")


def plot_vffsens(metric, ylab, out):
    """VFF at each learning rate against the MFF band, family mean over the whole run."""
    S = pd.read_csv(ce.R + f"vffsens_summary{T}.csv")
    S = S[(S.window == "whole") & (S.copula == "mean4")]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(ce.W, 5.2), sharex=True, sharey="row")
        for (sc, _), row in zip(SC, axs):
            for ax, e in zip(row, C.ESTIMATORS):
                vffsens_panel(ax, S[(S.metric == metric) & (S.scenario == sc) & (S.estimator == e)].set_index("strategy"),
                              e)
        h = [plt.Line2D([], [], color=ce.MUTED, lw=1.2, marker="o", ms=3), plt.Line2D([], [], color=ce.EDGE, lw=1.1, ls=(0, (4, 2))),
             plt.Line2D([], [], color=ce.REF, lw=0.7, ls=(0, (1, 1.5)))]
        scenario_grid(fig, axs, ylab, h, ["VFF (± 2 SE)", "MFF (± 2 SE band)", r"$\alpha=0.02$ (development)"],
                      r"VFF learning rate, $\alpha$ (log scale)", hspace=0.35)
        ce.save(fig, out.replace(".png", T + ".png"))


def main():
    os.makedirs(ce.OUT, exist_ok=True)
    plot_strategy_bars()
    plot_rmse_trajectories()
    plot_jump_trajectories()
    W = load_weights()
    for e in C.ESTIMATORS:
        plot_mff_weights(e, W)
        plot_fixed_share(e, W)
    plot_vffsens("rmse", RMSE_LABEL, "ch6_vffsens_rmse.png")
    plot_vffsens("h2_x1000", r"$H^2$ ($\times10^{-3}$)", "ch6_vffsens_h2.png")


if __name__ == "__main__":
    main()
