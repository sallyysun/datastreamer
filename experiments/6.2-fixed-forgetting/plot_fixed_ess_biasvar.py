"""Plot bias, variance and MSE across fixed memories under Gaussian drift."""
import sys
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
COMP = [("mse", "MSE", "#303030", "-", 1.1, "o", 3), ("bias2", "Squared bias", "#E69F00", (0, (4, 2)), 1.2, "^", 4),
        ("var", "Variance", "#0072B2", (0, (1, 1.3)), 1.4, "s", 5)]
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]


def load():
    d = pd.read_csv(ce.R + "fixed_ess_decomp.csv", dtype={"neff": str})
    d = d[(d.scenario == "drift") & (d.window == "whole") & (d.neff != "nocum")].copy()
    d["N"] = d.neff.astype(float)
    opt = pd.read_csv(ce.R + "fixed_ess_optimum.csv")
    return d, opt[opt.copula == "gaussian"].set_index("estimator")


def main():
    d, opt = load()
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.5), sharey=True)
        for ax, (e, title) in zip(axs, EST):
            g = d[d.estimator == e].sort_values("N").drop_duplicates("N")
            g = g[g.N <= 12000]
            coarse = g[g.grid != "fine"]   # the fine grid (200-600) is drawn as line only
            for q, lab, col, ls, lw, mk, z in COMP:
                ax.plot(g.N, 1e3 * g[q], color=col, ls=ls, lw=lw, zorder=z)
                ax.plot(coarse.N, 1e3 * coarse[q], ls="none", marker=mk, ms=1.8, mec=col, mfc=col, zorder=z)
                ax.plot([], [], color=col, ls=ls, lw=lw, marker=mk, ms=2.2, label=lab)
            ax.axvline(opt.loc[e, "optimum"], color="#9a9a9a", lw=0.6, zorder=1, label="Empirical optimum")
            ax.axvline(opt.loc[e, "approx"], color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=1, label="Approximation")
            ax.set_xscale("log")
            ax.set_xlim(20, 15000)
            ax.set_ylim(0, 20)
            ax.set_title(title)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Squared Error ($\times10^{-3}$)")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 1.0), handlelength=2.2, columnspacing=1.2)
        fig.text(0.53, 0.015, r"Effective Sample Size, $N_{\mathrm{eff}}$ (log scale)", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.8, bottom=0.2, wspace=0.12)
        out = ce.OUT + "ch6_drift_ess_bias_variance.png"
        fig.savefig(out)


if __name__ == "__main__":
    main()
