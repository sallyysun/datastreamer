"""Section 6.2 'Effective sample size following abrupt changes': RMSE of rho_S against N_eff before the jump, in the first
100 observations after it and in the next 400, one panel per estimator (RMPL, Counting Matrix, Hermite Series), Gaussian
copula. Data: results/fixed_ess_seeds[_fam].csv (probe_fixed_ess.cpp; RMPL on cumulative DataSketches KLL k = 256 margins;
60 seeds 42-101; forecast held before each tick). Windows (probe_fixed_ess): pre = t 500-2499, rec_0_100 = 2500-2599,
rec_100_500 = 2600-2999; RMSE per seed over the window, then the mean over seeds (as the other Chapter 6 figures).
Also checks the best memory per window in all four families and writes results/jump_ess_rmse_summary.csv.
Usage (repo root): python3 experiments/plot_jump_ess.py [relative]
  -> images/ch6/ch6_jump_ess_windows.png (relative: images/ch6/ch6_jump_ess_windows_relative.png)"""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402  (RC, layout of the approved constant / drift figures)

plt, core = ce.plt, ce.core
ARMS = ["39", "99", "199", "399", "1999", "nocum"]
LAB = ["39", "99", "199", "399", "1999", "No\nforgetting"]
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
WIN = [("pre", "Before jump", "#333333", "-", "o"), ("rec_0_100", "First 100 after jump", "#E69F00", (0, (4, 2)), "^"),
       ("rec_100_500", "Next 400 after jump", "#0072B2", (0, (1, 1.3)), "s")]
FAMS = [("gaussian", ""), ("clayton", "_clayton"), ("gumbel", "_gumbel"), ("t", "_t")]
OUTDIR = "images/ch6/"


def load(sfx):
    d = pd.read_csv(core.R + f"fixed_ess_seeds{sfx}.csv", dtype={"neff": str})
    d = d[(d.scenario == "jump") & (d.grid == "main")].copy()
    d["k"] = d.neff.map(lambda v: "nocum" if v == "nocum" else str(round(float(v))))
    return d


def summary():
    rows = []
    for f, s in FAMS:
        d = load(s)
        for e, _ in EST:
            for w in ["pre", "rec_0_100", "rec_100_500", "post", "whole"]:
                p = d[(d.estimator == e) & (d.window == w)].pivot(index="seed", columns="k", values="rmse")[ARMS] * 100
                mu, se = p.mean(), p.std(ddof=1) / math.sqrt(len(p))
                best = mu.idxmin()
                for a in ARMS:
                    rows.append(dict(copula=f, estimator=e, window=w, neff=a, rmse=mu[a], se=se[a], best=a == best))
    s = pd.DataFrame(rows)
    s.to_csv(core.R + "jump_ess_rmse_summary.csv", index=False)
    b = s[s.best].pivot_table(index=["copula", "estimator"], columns="window", values="neff", aggfunc="first")
    print(b[["pre", "rec_0_100", "rec_100_500", "post", "whole"]].to_string())
    for w in ["pre", "rec_0_100", "rec_100_500", "post", "whole"]:
        print(f"  {w:12s} best memory across the 12 cells: {b[w].value_counts().to_dict()}")
    return s


def figure(s, relative=False):
    g = s[s.copula == "gaussian"]
    x = np.array([0, 1, 2, 3, 4, 5.35])
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.6), sharey=True)
        for ax, (e, title) in zip(axs, EST):
            for w, lab, col, ls, mk in WIN:
                c = g[(g.estimator == e) & (g.window == w)].set_index("neff").loc[ARMS]
                y = c.rmse / c.rmse.min() if relative else c.rmse
                ax.plot(x[:5], y.values[:5], color=col, ls=ls, lw=1.3, marker=mk, ms=3.4, zorder=3, label=lab)
                ax.plot(x[5], y.values[5], marker="D", ms=4.2, color=col, mfc="white", mew=1.3, ls="none", zorder=4)
                k = ARMS.index(c.index[c.best.values][0])   # best memory in this window: a ring around its marker
                ax.plot(x[k], y.values[k], marker="o", ms=8, mfc="none", mec=col, mew=0.9, ls="none", zorder=5)
            ax.set_title(title)
            ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7)
            ax.set_xlim(-0.45, 6.0)
            if relative:
                ax.set_yscale("log"); ax.set_ylim(0.9, 20)
                ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"]); ax.minorticks_off()
            else:
                ax.set_ylim(0, 62)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0)
        axs[0].set_ylabel("RMSE / best in period (log scale)" if relative else r"Spearman RMSE ($\times10^{-2}$)")
        h, l = axs[0].get_legend_handles_labels()
        h.append(plt.Line2D([], [], marker="o", ms=7, mfc="none", mec="#52514e", ls="none")); l.append("Lowest RMSE in period")
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0),
                   handlelength=2.2, columnspacing=1.2)
        fig.text(0.53, 0.015, r"Effective Sample Size, $N_{\mathrm{eff}}$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.8, bottom=0.22, wspace=0.12)
        os.makedirs(OUTDIR, exist_ok=True)
        out = OUTDIR + ("ch6_jump_ess_windows_relative.png" if relative else "ch6_jump_ess_windows.png")
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    s = summary()
    figure(s, relative="relative" in sys.argv[1:])
