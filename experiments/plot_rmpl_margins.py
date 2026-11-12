"""RMPL margin sensitivity figures (style of ch6_fig_constant_ess): RMSE of rho_S against dependence N_eff for RMPL with
EW-500, cumulative-EW and oracle-PIT margins; constant (A) and drift (B); bias / variance under drift (C).
Data: results/memory_sweep_matched_rmpl2.csv (probe_memory, see analyze_rmpl_margins.py).
Usage (repo root): python3 experiments/plot_rmpl_margins.py -> notes/thesis/figures/matched/ch6_rmpl_margins_{constant,drift,biasvar}.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402  (RC, colours, paths of the constant-ESS figure)

plt, core = ce.plt, ce.core
CFG = [("rmpl_ew", "EW margins ($N_{\\mathrm{eff}}$ 500)", "#1baf7a", "-", "o"),
       ("rmpl_ewcum", "Cumulative EW margins", "#6250d6", (0, (4, 2)), "s"),
       ("rmpl_exact", "Oracle PIT", "#0b0b0b", (0, (1, 1.5)), "^")]
NOCUM = 1e9
d = pd.read_csv(core.R + "memory_sweep_matched_rmpl2.csv")
d = d[d.window == "all"]


def rmse_fig(sc, ylim, out):
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(core.W, 2.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.09], wspace=0.06)
        ax, axn = fig.add_subplot(gs[0]), None
        axn = fig.add_subplot(gs[1], sharey=ax)
        for f, lab, col, ls, mk in CFG:
            g = d[(d.scenario == sc) & (d.estimator == f)].sort_values("mem_ticks")
            fin, nc = g[g.mem_ticks < NOCUM], g[g.mem_ticks > NOCUM]
            ax.plot(fin.mem_ticks, 100 * fin.rmse, color=col, ls=ls, lw=1.4, marker=mk, ms=3.2, label=lab)
            axn.plot([0], 100 * nc.rmse.values, marker="D", ms=4.2, color=col, mfc="white", mew=1.3, ls="none")
        ax.set_xscale("log"); ax.set_xlim(20, 4e4); ax.set_ylim(0, ylim)
        ax.set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
        ax.set_xlabel(r"Dependence Effective Sample Size, $N_{\mathrm{eff}}$ (log scale)")
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="upper right" if sc == "constant" else "upper center", frameon=False, fontsize=7.5)
        axn.set_xlim(-0.6, 0.6); axn.set_xticks([0]); axn.set_xticklabels(["No\nforgetting"], fontsize=7)
        axn.tick_params(axis="y", labelleft=False, length=0); axn.tick_params(axis="x", length=0)
        axn.spines["left"].set_visible(False)
        axn.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); axn.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.2)
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def biasvar(out):
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 2, figsize=(core.W, 2.4), sharey=True)
        for ax, (q, title) in zip(axs, [("bias2", "Squared bias"), ("var", "Variance")]):
            for f, lab, col, ls, mk in CFG:
                g = d[(d.scenario == "drift") & (d.estimator == f) & (d.mem_ticks < 12000)].sort_values("mem_ticks")
                ax.plot(g.mem_ticks, 1e3 * g[q], color=col, ls=ls, lw=1.4, marker=mk, ms=3, label=lab)
            ax.set_xscale("log"); ax.set_title(title); ax.set_ylim(0, 12)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Error ($\times10^{-3}$)")
        axs[0].legend(loc="upper left", frameon=False, fontsize=7)
        fig.text(0.53, 0.015, r"Dependence Effective Sample Size, $N_{\mathrm{eff}}$ (log scale)", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.2, wspace=0.08)
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    rmse_fig("constant", 6, core.OUT + "ch6_rmpl_margins_constant.png")
    rmse_fig("drift", 21, core.OUT + "ch6_rmpl_margins_drift.png")
    biasvar(core.OUT + "ch6_rmpl_margins_biasvar.png")
