"""Section 6.2: Spearman RMSE against N_eff for the five fixed retention factors and no forgetting (RMPL, Counting Matrix,
Hermite Series; Gaussian copula; constant and drift paths; 60 seeds 42-101), from results/fixed_ess_seeds.csv (probe_fixed_ess).
Usage: python3 plot_constant_ess.py -> images/ch6_constant_ess_rmse.png, images/ch6_drift_ess_rmse.png"""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
ARMS = [f"fixed_{m}" for m in C.MEMORIES] + ["nocum"]
LAB = [str(m) for m in C.MEMORIES] + ["No\nforgetting"]
EST = [(C.EST_COLOUR_KEY[e], C.EST_LABEL[e]) for e in C.ESTIMATORS]
SC = 100   # RMSE x 10^-2


def load(f):
    """results/fixed_ess_seeds[_fam].csv (probe_fixed_ess; RMPL on cumulative DataSketches KLL k = 256 margins)."""
    d = pd.read_csv(core.R + f"fixed_ess_seeds{core.sfx(f)}.csv", dtype={"neff": str})
    d = d[(d.grid == "main") & (d.window == "whole")].copy()
    d["component"] = d.estimator.replace({"rmpl": "mle"})
    d["method"] = d.neff.map(lambda v: "nocum" if v == "nocum" else f"fixed_{round(float(v))}")
    return d


COL, INK, GRID, RC = ce.COL, ce.INK, ce.GRID, ce.RC


def figure(scenario="constant", out=core.OUT + "ch6_constant_ess_rmse.png", ylim=13.5, label_nocum=False, est=None):
    """constant: value labels at N_eff 1999 and no forgetting. drift: the minimum RMSE labelled (and, with
    label_nocum, the no-forgetting RMSE)."""
    d = load("gaussian"); d = d[d.scenario == scenario]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.35), sharey=True)
        x = np.array([0, 1, 2, 3, 4, 5.35])   # a gap before no forgetting: it is not a point on the N_eff scale
        for ax, (e, title) in zip(axs, est or EST):
            p = d[d.component == e].pivot(index="seed", columns="method", values="rmse") * SC
            mu = p[ARMS].mean(); se = p[ARMS].std(ddof=1) / math.sqrt(len(p))
            col = COL[e]
            ax.plot(x[:5], mu[ARMS[:5]], color=col, lw=1.4, zorder=3)
            ax.errorbar(x[:5], mu[ARMS[:5]], yerr=2 * se[ARMS[:5]], fmt="o", ms=3.8, color=col, lw=1, capsize=0, zorder=4)
            ax.errorbar(x[5], mu["nocum"], yerr=2 * se["nocum"], fmt="D", ms=4.2, color=col, mfc="white", mew=1.3, lw=1,
                        capsize=0, zorder=4)
            if scenario == "constant":
                for a, xi in [("fixed_1999", x[4]), ("nocum", x[5])]:
                    ax.text(xi, mu[a] + 0.75, f"{mu[a]:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
            else:
                k = int(np.argmin(mu[ARMS[:5]].values))
                print(f"  {title} {scenario}: minimum {mu[ARMS[k]]:.2f} at N_eff {ARMS[k][6:]}")
                ax.text(x[k], mu[ARMS[k]] - 0.75 * ylim / 13.5, f"{mu[ARMS[k]]:.2f}", ha="center", va="top",
                        fontsize=6.8, color=INK)
                if label_nocum:
                    ax.text(x[5], mu["nocum"] + 0.75 * ylim / 13.5, f"{mu['nocum']:.2f}", ha="center", va="bottom",
                            fontsize=6.8, color=INK)
            m, s = core.paired(p.nocum, p.fixed_1999)
            if scenario == "constant": print(f"  {title}: no forgetting - 1999 = {m:+.2f} ± {2*s:.2f} (x1e-2, paired 2 SE)")
            ax.set_title(title)
            ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7)
            ax.set_xlim(-0.45, 6.0); ax.set_ylim(0, ylim)
            ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0)
        axs[0].set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
        fig.text(0.53, 0.015, r"Effective Sample Size, $N_{\mathrm{eff}}$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.9, bottom=0.24, wspace=0.12)
        fig.savefig(out)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure("constant", core.OUT + "ch6_constant_ess_rmse.png")
    figure("drift", core.OUT + "ch6_drift_ess_rmse.png", ylim=21, label_nocum=True)
