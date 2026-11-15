"""Chapter 6.3 MFF regret over time (exploratory). results/regret_paths.csv (probe_strategies REGRET=1; analyze_regret.py).
Log loss, eta = 1, the mixture forecaster -log sum_k w_k c_k (the one the Chapter 5 bounds cover). Regret at n =
cumulative MFF loss - the comparator's cumulative loss over the scored ticks up to n, the comparator re-optimised at each n:
row 1 best single arm (static), row 2 best arm sequence with at most one switch. Mean over 60 seeds, band ± 2 SE.
Ordinary Hedge (delta = 0) vs fixed share (delta = 1e-4).
Usage (repo root): python3 experiments/plot_mff_regret.py [est] -> images/ch6/ch6_mff_regret_<est>.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
EL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
STR = [(0.0, r"Ordinary Hedge ($\delta=0$)", "#7a7a7a", (0, (4, 2))), (1e-4, r"Fixed share ($\delta=10^{-4}$)", "#C2185B", "-")]
ROWS = [(0, "Static regret\n(nats)"), (1, "Switching regret,\n" + r"$\leq$1 switch (nats)")]
p = pd.read_csv(core.R + "regret_paths.csv")
p = p[p.loss == "mix"]


def fig(e):
    with plt.rc_context(ce.RC):
        f, axs = plt.subplots(2, 3, figsize=(core.W, 3.6), sharex=True, sharey="row")
        for ri, (m, yl) in enumerate(ROWS):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]
                for d, lab, col, ls in STR:
                    g = p[(p.estimator == e) & (p.scenario == sc) & np.isclose(p.delta, d) & (p.m == m)].sort_values("t")
                    se = 2 * g.sd / np.sqrt(60)
                    ax.fill_between(g.t, g["mean"] - se, g["mean"] + se, color=col, alpha=0.18, lw=0, zorder=2)
                    ax.plot(g.t, g["mean"], color=col, ls=ls, lw=1.1, zorder=3)
                ax.axhline(0, color="#333333", lw=0.6, zorder=1)
                if sc == "jump": ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
                ax.set_xlim(0, 5000); ax.set_xticks([1000, 2500, 4000])
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if ri == 0: ax.set_title(scl)
            axs[ri, 0].set_ylabel(yl)
        h = [plt.Line2D([], [], color=c, ls=s, lw=1.1) for _, _, c, s in STR]
        f.legend(h, [l for _, l, _, _ in STR], loc="upper center", ncol=2, frameon=False, fontsize=7,
                 bbox_to_anchor=(0.54, 1.0), title=f"{EL[e]} estimator", title_fontsize=7, handlelength=2.2)
        f.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=8)
        f.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.12, hspace=0.15, wspace=0.08)
        out = f"images/ch6/ch6_mff_regret_{e}.png"
        f.savefig(out); plt.close(f)
    print("wrote", out)


for e in (sys.argv[1:] or ["rmpl", "xiao", "hermite"]): fig(e)
