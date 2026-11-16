"""Section 6.2/6.7 trade-off figure at n = 15,000. (a) Box plots of the percentage reduction in test Spearman RMSE relative to
each estimator's own 16 KiB configuration, at 64 and 256 KiB; each box summarises the 12 generating family x dependence scenario
conditions (variation across conditions, not Monte Carlo uncertainty); per condition 100 * (mean RMSE_16 - mean RMSE_b) /
mean RMSE_16 over the 100 paired test seeds, configurations chosen on validation seeds (results/budget_tradeoff_summary.csv).
(b) Processing time per observation against retained state for every tested approximation size, n = 15,000 streams
(probe_budget time, gain 0.02, single thread, mean of 5 runs; results/budget_time_table_n15k.csv).
Usage (from experiments/): python3 plot_budget_tradeoff_n15k.py -> ../images/ch6/6.2/ch6_budget_tradeoff_n15k.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
E = [("mpl_dskll", "RMPL", ce.COL["mle"], "^"), ("xiao", "Counting Matrix", ce.COL["xiao"], "o"),
     ("hermite", "Hermite Series", ce.COL["hermite"], "s")]
P = pd.read_csv("results/budget_tradeoff_summary.csv")
P = P[P.n == "n = 15,000"]
T = pd.read_csv("results/budget_time_table_n15k.csv")
with plt.rc_context(ce.RC):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(core.W, 2.5), gridspec_kw=dict(width_ratios=[1, 1.05]))
    pos, data, cols, alphas = [], [], [], []
    for i, (e, _, c, _) in enumerate(E):
        for j, b in enumerate((64, 256)):
            pos.append(i + (j - 0.5) * 0.36); data.append(P[(P.estimator == e) & (P.budget == b)].pct.values)
            cols.append(c); alphas.append(0.35 if b == 64 else 0.85)
    bp = axa.boxplot(data, positions=pos, widths=0.3, whis=1.5, patch_artist=True, showfliers=True,
                     medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                     capprops=dict(color="#555555", lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#555555", mew=0.6))
    for p, c, a in zip(bp["boxes"], cols, alphas):
        p.set_facecolor(c); p.set_alpha(a); p.set_edgecolor("#444444"); p.set_linewidth(0.7)
    axa.axhline(0, color="#222222", lw=0.6, zorder=0)
    axa.set_xticks(range(3)); axa.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"], fontsize=7); axa.tick_params(axis="x", length=0)
    axa.set_xlim(-0.6, 2.6); axa.set_ylim(-1, 10)
    axa.grid(True, axis="y", color=ce.GRID, lw=0.5); axa.set_axisbelow(True)
    axa.set_ylabel("RMSE reduction vs 16 KiB (%)", fontsize=7.5)
    axa.set_title("(a) Accuracy gain from more storage", fontsize=8, pad=3, loc="left")
    axa.legend([plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.35), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.85)],
               ["64 KiB", "256 KiB"], frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
    for e, lab, c, m in E:
        x = T[T.estimator == e]
        axb.plot(x.kib, x.us, color=c, marker=m, ms=3, lw=1.1, label=lab)
    axb.set_xscale("log", base=2); axb.set_yscale("log", base=2)
    axb.set_xticks([1, 4, 16, 64, 256]); axb.set_xticklabels(["1", "4", "16", "64", "256"])
    axb.set_yticks([0.25, 1, 4, 16, 64]); axb.set_yticklabels(["0.25", "1", "4", "16", "64"])
    for b in (16, 64, 256): axb.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
    axb.grid(True, color=ce.GRID, lw=0.5); axb.set_axisbelow(True)
    axb.set_xlabel("Retained state (KiB)", fontsize=7.5); axb.set_ylabel("Time per observation (µs)", fontsize=7.5)
    axb.set_title("(b) Processing time", fontsize=8, pad=3, loc="left")
    axb.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.18, wspace=0.3)
    fig.savefig("../images/ch6/6.2/ch6_budget_tradeoff_n15k.png"); plt.close(fig)
print(P.groupby(["estimator", "budget"]).pct.describe(percentiles=[.25, .5, .75]).round(2).to_string())
