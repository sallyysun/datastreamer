"""Exploratory: MSE against N_eff under Gaussian gradual drift, normalised by each estimator's minimum, to show how flat the
MSE is near the optimum. results/fixed_ess_decomp.csv (probe_fixed_ess; 60 seeds; forecasts before each observation;
whole window). Markers: the 20-point sweep (25-30000); diamond: N_eff = 399 (main grid). Normalisation: MSE / the lowest
MSE over all evaluated memories (sweep, fine 200-600 and main grids). Solid line: fitted optimum (log-quadratic,
results/fixed_ess_optimum.csv); dashed line: the common N_eff = 399. Log-log axes.
Usage (repo root): python3 experiments/plot_ess_flatness.py -> images/ch6/explore_ess_flatness.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
EST = [("rmpl", "RMPL", "mle"), ("xiao", "Counting Matrix", "xiao"), ("hermite", "Hermite Series", "hermite")]
d = pd.read_csv(core.R + "fixed_ess_decomp.csv", dtype={"neff": str})
d = d[(d.scenario == "drift") & (d.window == "whole") & (d.neff != "nocum")].copy()
d["N"] = d.neff.astype(float)
opt = pd.read_csv(core.R + "fixed_ess_optimum.csv")
opt = opt[opt.copula == "gaussian"].set_index("estimator")

def draw(zoom, out):
  with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.6), sharey=True)
    top = 0
    for ax, (e, el, ck) in zip(axs, EST):
        g = d[d.estimator == e].drop_duplicates("N")
        mmin = g.mse.min()
        sw = g[g.grid == "sweep"].sort_values("N")
        if zoom: sw = g[(g.grid != "fine") & (g.N >= 100) & (g.N <= 2000)].sort_values("N")   # sweep + main points in range
        col = ce.COL[ck]
        ax.plot(sw.N, sw.mse / mmin, color=col, lw=1.2, marker="o", ms=3, zorder=3)
        top = max(top, (sw.mse / mmin).max())
        r399 = g[np.isclose(g.N, 399)].mse.iloc[0] / mmin
        ax.plot([399], [r399], marker="D", ms=4.5, color=col, mfc="white", mew=1.2, ls="none", zorder=4)
        no = opt.loc[e, "optimum"]
        ax.axvline(no, color="#333333", lw=0.8, zorder=2)
        ax.axvline(399, color="#333333", lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_title(f"{el}\n" + r"optimum %.0f; RMSE at 399 +%.1f%%" % (no, 100 * (np.sqrt(r399) - 1)), fontsize=6.6)
        ax.set_xscale("log")
        if zoom:
            ax.set_xlim(90, 2200); ax.set_xticks([100, 200, 400, 1000, 2000]); ax.set_xticklabels(["100", "200", "400", "1000", "2000"])
        else:
            ax.set_yscale("log"); ax.set_xlim(20, 4e4)
            ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"])
            ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["$10^2$", "$10^3$", "$10^4$"])
        ax.minorticks_off()
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    axs[0].set_ylim((0.9, 1.08 * top) if zoom else (0.9, 1.15 * top))
    axs[0].set_ylabel("MSE / minimum MSE")
    h = [plt.Line2D([], [], color="#6b6b6b", lw=1.2, marker="o", ms=3), plt.Line2D([], [], color="#6b6b6b", marker="D", ms=4.5, mfc="white", ls="none"),
         plt.Line2D([], [], color="#333333", lw=0.8), plt.Line2D([], [], color="#333333", lw=0.8, ls=(0, (3, 2)))]
    fig.legend(h, ["Simulated MSE", r"$N_{\mathrm{eff}}=399$", "Fitted optimum", r"Common $N_{\mathrm{eff}}=399$"],
               loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=1.8)
    fig.text(0.53, 0.015, r"Effective Sample Size, $N_{\mathrm{eff}}$ (log scale)", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.76, bottom=0.2, wspace=0.1)
    fig.savefig("images/ch6/" + out); plt.close(fig)
  print("wrote images/ch6/" + out)


draw(False, "explore_ess_flatness.png")        # full 20-point sweep, log-log
draw(True, "explore_ess_flatness_zoom.png")    # N_eff 100-2000, linear y
