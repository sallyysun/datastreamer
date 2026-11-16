"""Section 6.7 standalone figures.
(1) ch6_budget_processing_time.png: processing time per observation (update + estimate, one thread, gain 0.02, mean of 5 runs,
    n = 15,000 streams; results/budget_time_table_n15k.csv; the n = 5000 version goes to appendix/) against retained state, every tested approximation size.
(2) ch6_budget_accuracy_gain_pct.png: box plots of the percentage reduction in test Spearman RMSE relative to each estimator's
    own validation-selected 16 KiB configuration, at 64 and 256 KiB; (a) n = 5000, (b) n = 15,000. Each box = the 12 generating
    family x dependence scenario conditions (variation across conditions); per condition 100 (mean RMSE_16 - mean RMSE_b) /
    mean RMSE_16 over the 100 paired test seeds (results/budget_tradeoff_summary.csv, from plot_budget_tradeoff.py).
Usage (from experiments/): python3 plot_budget_67.py"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
OUT = "../images/ch6/6.7/"
import os
os.makedirs(OUT + "appendix", exist_ok=True)
E = [("mpl_dskll", "RMPL", ce.COL["mle"], "^"), ("xiao", "Counting Matrix", ce.COL["xiao"], "o"),
     ("hermite", "Hermite Series", ce.COL["hermite"], "s")]

for tfile, outname in [("results/budget_time_table_n15k.csv", OUT + "ch6_budget_processing_time.png"),            # main: n = 15,000
                       ("results/budget_time_table.csv", OUT + "appendix/ch6_budget_processing_time_n5000.png")]:  # appendix
    T = pd.read_csv(tfile)
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(core.W * 0.62, 2.5))
        for e, lab, c, m in E:
            x = T[T.estimator == e]
            ax.plot(x.kib, x.us, color=c, marker=m, ms=3, lw=1.1, label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
        ax.set_yticks([0.25, 1, 4, 16, 64]); ax.set_yticklabels(["0.25", "1", "4", "16", "64"])
        for b in (16, 64, 256): ax.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
        ax.grid(True, color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("Retained state (KiB)", fontsize=7.5); ax.set_ylabel("Time per observation (µs)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left")
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.17)
        fig.savefig(outname); plt.close(fig)

P = pd.read_csv("results/budget_tradeoff_summary.csv")
lo, hi = P.pct.min(), P.pct.max()
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 2, figsize=(core.W, 2.45), sharey=True)
    for ax, (nl, t) in zip(axs, [("n = 5000", "(a) $n = 5000$"), ("n = 15,000", "(b) $n = 15{,}000$")]):
        pos, data, cols, alphas = [], [], [], []
        for i, (e, _, c, _) in enumerate(E):
            for j, b in enumerate((64, 256)):
                pos.append(i + (j - 0.5) * 0.36); data.append(P[(P.n == nl) & (P.estimator == e) & (P.budget == b)].pct.values)
                cols.append(c); alphas.append(0.35 if b == 64 else 0.85)
        bp = ax.boxplot(data, positions=pos, widths=0.3, whis=1.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                        capprops=dict(color="#555555", lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#555555", mew=0.6))
        for p, c, a in zip(bp["boxes"], cols, alphas):
            p.set_facecolor(c); p.set_alpha(a); p.set_edgecolor("#444444"); p.set_linewidth(0.7)
        ax.axhline(0, color="#222222", lw=0.6, zorder=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"], fontsize=7); ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(np.floor(lo) - 0.2, np.ceil(hi) + 0.3)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(t, fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("RMSE reduction vs 16 KiB (%)", fontsize=7.5)
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.35), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.85)],
                  ["64 KiB", "256 KiB"], frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.13, wspace=0.06)
    fig.savefig(OUT + "ch6_budget_accuracy_gain_pct.png"); plt.close(fig)
print(f"y range {lo:.2f} to {hi:.2f}")
print(P.groupby(["n", "estimator", "budget"]).pct.describe(percentiles=[.25, .5, .75])[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
print(P.groupby(["n", "budget"]).pct.median().round(2).to_string())
