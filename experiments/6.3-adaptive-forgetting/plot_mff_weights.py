"""Section 6.3.3 (exploratory): MFF weight adaptation, Gaussian copula, from results/strategies_weights.csv
(probe_strategies WEIGHTS=1: the main MFF, log loss, eta = 1, delta = 1e-4, over the five fixed arms N_eff 39-1999, and
plain Hedge delta = 0 on the same arms and streams; KLL margins; 60 seeds 42-101; the delta = 1e-4 RMSE reproduces
strategies_seeds exactly). Weights are those held before each observation (the ones the forecast uses), averaged over
seeds at each n; the seed-mean weights sum to one. Weight-averaged candidate ESS: sum_k w_k N_eff,k from those mean
weights (linear, so equal to the seed mean of the per-seed value). It is not the ESS of the combined estimator.
  ch6_mff_weights_<est>.png      row 1 stacked weights, row 2 weight-averaged candidate ESS (delta = 1e-4)
  ch6_mff_fixedshare_<est>.png   fixed share vs plain Hedge: row 1 weight-averaged candidate ESS, row 2 pointwise RMSE
                                 (sqrt of the seed-mean squared error, MSE smoothed over a centred 25-observation window
                                 within each regime)
Usage: python3 plot_mff_weights.py -> images/"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
CMAP = plt.get_cmap("plasma")
MEM = [(m, CMAP(C.MEMORY_CMAP_POS[m])) for m in C.MEMORIES]
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
JUMP, T0, T1 = 2500, 500, 5000
FS_COL, HEDGE_COL = "#C2185B", "#7a7a7a"
W = pd.read_csv(core.R + "strategies_weights.csv")
W = W[(W.t >= T0) & (W.t < T1)]
WC = [f"w{m}" for m, _ in MEM]


def get(sc, e, delta):
    g = W[(W.scenario == sc) & (W.estimator == e) & np.isclose(W.delta, delta)].sort_values("t").copy()
    g[WC] = g[WC].div(g[WC].sum(axis=1), axis=0)          # renormalise the float-rounded mean (sums within 2e-6 of 1)
    g["nbar"] = g[WC].values @ np.array([m for m, _ in MEM], float)
    return g


def regimes(g, sc):
    return [g[g.t < JUMP], g[g.t >= JUMP]] if sc == "jump" else [g]


def frame(ax, sc):
    ax.set_xlim(T0, T1); ax.set_xticks([1000, 2500, 4000])
    if sc == "jump": ax.axvline(JUMP, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=6)


def nbar_axis(ax):
    ax.set_yscale("log"); ax.set_ylim(30, 2600)
    ax.set_yticks([m for m, _ in MEM]); ax.set_yticklabels([str(m) for m, _ in MEM]); ax.minorticks_off()
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def weights_fig(e, el):
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(core.W, 3.1), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
        for j, (sc, scl) in enumerate(SC):
            g = get(sc, e, 1e-4)
            axs[0, j].stackplot(g.t, *[g[c] for c in WC], colors=[c for _, c in MEM], lw=0, zorder=2)
            axs[0, j].set_ylim(0, 1); axs[0, j].set_title(scl)
            for p in regimes(g, sc): axs[1, j].plot(p.t, p.nbar, color="#222222", lw=1.1, zorder=4)
            nbar_axis(axs[1, j])
            for ax in axs[:, j]: frame(ax, sc)
        axs[0, 0].set_ylabel("Mean MFF weight")
        axs[1, 0].set_ylabel("Weight-averaged\ncandidate " + r"$N_{\mathrm{eff}}$")
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MEM]
        fig.legend(h, [r"$N_{\mathrm{eff}}=%d$" % m for m, _ in MEM], loc="upper center", ncol=5, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 1.0), handlelength=1.2, title=f"{el} estimator",
                   title_fontsize=7)
        fig.text(0.53, 0.01, "Observation number, $n$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.12, right=0.985, top=1 - 0.665 / 3.1, bottom=0.42 / 3.1, hspace=0.14, wspace=0.08)
        out = f"images/ch6_mff_weights_{e}.png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def ablation_fig(e, el, scen, out):
    """Columns: plain Hedge (delta = 0) | fixed share (delta = 1e-4); per scenario a stacked-weight row and an N-bar row."""
    rows = 2 * len(scen)
    with plt.rc_context(ce.RC):
        H = 1.05 * rows + 0.85
        fig, axs = plt.subplots(rows, 2, figsize=(core.W * 0.8, H), sharex=True, gridspec_kw={"height_ratios": [1.35, 1] * len(scen)})
        for c, (delta, cl) in enumerate([(0.0, r"Ordinary Hedge ($\delta=0$)"), (1e-4, r"Fixed share ($\delta=10^{-4}$)")]):
            for si, (sc, scl) in enumerate(scen):
                g = get(sc, e, delta)
                a0, a1 = axs[2 * si, c], axs[2 * si + 1, c]
                a0.stackplot(g.t, *[g[k] for k in WC], colors=[col for _, col in MEM], lw=0, zorder=2)
                a0.set_ylim(0, 1); a0.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                for p in regimes(g, sc): a1.plot(p.t, p.nbar, color="#222222", lw=1.1, zorder=4)
                nbar_axis(a1)
                for ax in (a0, a1): frame(ax, sc)
                if c == 0:
                    a0.set_ylabel(f"{scl}\nmean weight")
                    a1.set_ylabel("Weight-averaged\ncandidate " + r"$N_{\mathrm{eff}}$")
                else:
                    a0.set_yticklabels([]); a1.set_yticklabels([])
            axs[0, c].set_title(cl)
        h = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col in MEM]
        fig.legend(h, [r"$N_{\mathrm{eff}}=%d$" % m for m, _ in MEM], loc="upper center", ncol=5, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.55, 1.0), handlelength=1.2, title=f"{el} estimator", title_fontsize=7)
        fig.text(0.56, 0.008, "Observation number, $n$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.15, right=0.98, top=1 - 0.66 / H, bottom=0.45 / H, hspace=0.16, wspace=0.06)
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


for e, el in EST:
    weights_fig(e, el)
    ablation_fig(e, el, SC[2:], f"images/ch6_mff_fixedshare_jump_{e}.png")
