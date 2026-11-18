"""Section 6.2 figures: fixed-memory RMSE, its bias-variance split, and recovery after the jump."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
RMSE_LABEL = r"Spearman RMSE ($\times10^{-2}$)"
NEFF_LABEL = r"Effective Sample Size, $N_{\mathrm{eff}}$"
FIXED = [f"fixed_{m}" for m in C.MEMORIES]
ARMS = FIXED + ["nocum"]
ARM_LABELS = [str(m) for m in C.MEMORIES] + ["No\nforgetting"]


def estimator_colour(estimator):
    return ce.COL[C.EST_COLOUR_KEY[estimator]]


def estimator_panels(height):
    """One panel per estimator, sharing the y axis."""
    fig, axs = plt.subplots(1, 3, figsize=(ce.W, height), sharey=True)
    for ax, estimator in zip(axs, C.ESTIMATORS):
        ax.set_title(C.EST_LABEL[estimator])
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    return fig, axs


def finish_panels(fig, axs, out, ylabel, xlabel, top, bottom, legend=None):
    axs[0].set_ylabel(ylabel)
    if legend is not None:
        fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 1.0), **legend)
    fig.text(0.53, 0.015, xlabel, ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.975, top=top, bottom=bottom, wspace=0.12)
    fig.savefig(ce.OUT + out)
    plt.close(fig)


def label_point(ax, x, y, value, offset, above):
    ax.text(x, y + offset if above else y - offset, f"{value:.2f}", ha="center", va="bottom" if above else "top",
            fontsize=6.8, color=ce.INK)


def whole_window_rmse(scenario):
    d = pd.read_csv(ce.R + "fixed_ess_seeds.csv", dtype={"neff": str})
    d = d[(d.grid == "main") & (d.window == "whole") & (d.scenario == scenario)].copy()
    d["method"] = d.neff.map(lambda v: "nocum" if v == "nocum" else f"fixed_{round(float(v))}")
    return d


def plot_memory_rmse(scenario, out, ylim, label_nocum=False):
    """Gaussian RMSE at each fixed memory and with no forgetting."""
    d = whole_window_rmse(scenario)
    with plt.rc_context(ce.RC):
        fig, axs = estimator_panels(2.35)
        x = np.array([0, 1, 2, 3, 4, 5.35])   # a gap before no forgetting: it is not a point on the N_eff scale
        offset = 0.75 * ylim / 13.5
        for ax, estimator in zip(axs, C.ESTIMATORS):
            p = d[d.estimator == estimator].pivot(index="seed", columns="method", values="rmse") * 100
            mu, se = p[ARMS].mean(), standard_error(p[ARMS])
            col = estimator_colour(estimator)
            ax.plot(x[:5], mu[FIXED], color=col, lw=1.4, zorder=3)
            ax.errorbar(x[:5], mu[FIXED], yerr=2 * se[FIXED], fmt="o", ms=3.8, color=col, lw=1, capsize=0, zorder=4)
            ax.errorbar(x[5], mu["nocum"], yerr=2 * se["nocum"], fmt="D", ms=4.2, color=col, mfc="white", mew=1.3, lw=1,
                        capsize=0, zorder=4)
            if scenario == "constant":
                for arm, xi in [("fixed_1999", x[4]), ("nocum", x[5])]:
                    label_point(ax, xi, mu[arm], mu[arm], 0.75, above=True)
            else:
                k = int(np.argmin(mu[FIXED].values))
                label_point(ax, x[k], mu[FIXED[k]], mu[FIXED[k]], offset, above=False)
                if label_nocum:
                    label_point(ax, x[5], mu["nocum"], mu["nocum"], offset, above=True)
            ax.set_xticks(x)
            ax.set_xticklabels(ARM_LABELS, fontsize=7)
            ax.set_xlim(-0.45, 6.0)
            ax.set_ylim(0, ylim)
            ax.tick_params(axis="x", length=0)
        finish_panels(fig, axs, out, RMSE_LABEL, NEFF_LABEL, top=0.9, bottom=0.24)


def plot_drift_bias_variance():
    """MSE, squared bias and variance across fixed memories under Gaussian drift."""
    d = pd.read_csv(ce.R + "fixed_ess_decomp.csv", dtype={"neff": str})
    d = d[(d.scenario == "drift") & (d.window == "whole") & (d.neff != "nocum")].copy()
    d["N"] = d.neff.astype(float)
    opt = pd.read_csv(ce.R + "fixed_ess_optimum.csv")
    opt = opt[opt.copula == "gaussian"].set_index("estimator")
    components = [("mse", "MSE", "#303030", "-", 1.1, "o", 3),
                  ("bias2", "Squared bias", "#E69F00", (0, (4, 2)), 1.2, "^", 4),
                  ("var", "Variance", "#0072B2", (0, (1, 1.3)), 1.4, "s", 5)]
    with plt.rc_context(ce.RC):
        fig, axs = estimator_panels(2.5)
        for ax, estimator in zip(axs, C.ESTIMATORS):
            g = d[d.estimator == estimator].sort_values("N").drop_duplicates("N")
            g = g[g.N <= 12000]
            coarse = g[g.grid != "fine"]   # the fine grid (200-600) is drawn as line only
            for q, lab, col, ls, lw, mk, z in components:
                ax.plot(g.N, 1e3 * g[q], color=col, ls=ls, lw=lw, zorder=z)
                ax.plot(coarse.N, 1e3 * coarse[q], ls="none", marker=mk, ms=1.8, mec=col, mfc=col, zorder=z)
                ax.plot([], [], color=col, ls=ls, lw=lw, marker=mk, ms=2.2, label=lab)
            ax.axvline(opt.loc[estimator, "optimum"], color="#9a9a9a", lw=0.6, zorder=1, label="Empirical optimum")
            ax.axvline(opt.loc[estimator, "approx"], color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=1, label="Approximation")
            ax.set_xscale("log")
            ax.set_xlim(20, 15000)
            ax.set_ylim(0, 20)
        finish_panels(fig, axs, "ch6_drift_ess_bias_variance.png", r"Squared Error ($\times10^{-3}$)",
                      NEFF_LABEL + " (log scale)", top=0.8, bottom=0.2,
                      legend=dict(ncol=5, handlelength=2.2, columnspacing=1.2))


def plot_jump_rmse(before=300, after=500):
    """Gaussian RMSE by tick around the jump, one line per memory."""
    cmap = plt.get_cmap("plasma")
    memories = [(f"{m}.000000", str(m), cmap(C.MEMORY_CMAP_POS[m]), "-") for m in C.MEMORIES] + [
        ("nocum", "No forgetting", "#0b0b0b", (0, (3, 1.6)))]
    d = pd.read_csv(ce.R + "fixed_ess_jumppath.csv", dtype={"neff": str})
    d = d[(d.t >= C.CHANGE - before) & (d.t <= C.CHANGE + after)]
    with plt.rc_context(ce.RC):
        fig, axs = estimator_panels(2.6)
        for ax, estimator in zip(axs, C.ESTIMATORS):
            for neff, lab, col, ls in memories:
                g = d[(d.estimator == estimator) & (d.neff == neff)].sort_values("t")
                ax.plot(g.t - C.CHANGE, 100 * np.sqrt(g.mse), color=col, ls=ls, lw=0.9, label=lab, zorder=3)
            ax.axvline(0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=2)
            ax.set_xlim(-before, after)
            ax.set_ylim(0, 65)
            ax.set_xticks([-250, 0, 250, 500])
        finish_panels(fig, axs, "ch6_jump_ess_rmse.png", RMSE_LABEL, "Observations relative to the jump", top=0.78,
                      bottom=0.18, legend=dict(ncol=6, handlelength=2.0, columnspacing=1.1, title=r"$N_{\mathrm{eff}}$",
                                               title_fontsize=7))


def plot_jump_recovery():
    """Empirical and predicted 90% recovery times by memory."""
    r = pd.read_csv(ce.R + "jump_recovery.csv")
    r = r[(r.copula == "gaussian") & (r.neff != "No forgetting")].copy()
    r["N"] = r.neff.astype(float)
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.62, 2.6))
        nn = np.geomspace(30, 2600, 200)
        ax.plot(nn, np.log(0.1) / np.log(1 - 2 / (nn + 1)), color="#9a9a9a", lw=0.9, ls=(0, (4, 2)), zorder=2,
                label="Theoretical")
        for estimator, off in zip(C.ESTIMATORS, [0.94, 1.0, 1.06]):   # offsets keep overlapping markers visible
            g = r[r.estimator == estimator].sort_values("N")
            ax.plot(g.N * off, g.recovery, ls="none", marker=C.EST_MARKER[estimator], ms=4.5,
                    color=estimator_colour(estimator), label=C.EST_LABEL[estimator], zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(C.MEMORIES)
        ax.set_xticklabels([str(m) for m in C.MEMORIES])
        ax.set_yticks([50, 100, 200, 500, 1000, 2000])
        ax.set_yticklabels(["50", "100", "200", "500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_xlim(30, 2700)
        ax.set_ylim(35, 3000)
        ax.set_xlabel(r"Effective sample size, $N_{\mathrm{eff}}$")
        ax.set_ylabel("90% recovery time (observations)")
        ax.grid(True, color=ce.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.3, labelspacing=0.35)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.17)
        fig.savefig(ce.OUT + "ch6_jump_recovery_time.png")
        plt.close(fig)


def main():
    plot_memory_rmse("constant", "ch6_constant_ess_rmse.png", ylim=13.5)
    plot_memory_rmse("drift", "ch6_drift_ess_rmse.png", ylim=21, label_nocum=True)
    plot_drift_bias_variance()
    plot_jump_rmse()
    plot_jump_recovery()


if __name__ == "__main__":
    main()
