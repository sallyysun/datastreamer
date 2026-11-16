"""Section 6.2: test Spearman RMSE by estimator and storage budget. Per test seed, average over the four families of the RMSE at
each family's validation-selected configuration; bars = mean over the 100 test seeds, error bars ±2 SE over seeds.
Source: results/budget_audit_familyavg.csv (audit_budget.py).
Usage (from experiments/): python3 plot_budget_rmse_bars.py -> ../images/ch6/6.2/ch6_budget_rmse_bars.png"""
import os
SUF = os.environ.get("BUDGET_SET", "")          # "" = n = 5000 (extended grid); "_n15k" = n = 15,000
SRC = SUF if SUF else "_ext"

import sys
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
FA = pd.read_csv(f"results/budget_audit_familyavg{SUF}.csv")
E = [("mpl_dskll", ce.COL["mle"]), ("xiao", ce.COL["xiao"]), ("hermite", ce.COL["hermite"])]
SC = [("constant", "(a) Constant"), ("drift", "(b) Gradual drift"), ("jump", "(c) Abrupt jump")]
B = [(16, 0.35), (64, 0.65), (256, 1.0)]
bw = 0.27
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.3))
    for ax, (s, t) in zip(axs, SC):
        for i, (e, col) in enumerate(E):
            for j, (b, a) in enumerate(B):
                r = FA[(FA.scenario == s) & (FA.estimator == e) & (FA.budget_kib == b)].iloc[0]
                x = i + (j - 1) * bw
                ax.bar(x, r["mean"], width=bw * 0.92, color=col, alpha=a, zorder=3)
                ax.errorbar(x, r["mean"], yerr=2 * r.se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
        ax.set_xticks(range(3)); ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"], fontsize=7); ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, None)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(t, fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("Spearman RMSE ($\\times10^{-2}$)", fontsize=7.5)
    h = [plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=a) for _, a in B]
    fig.legend(h, [f"{b} KiB" for b, _ in B], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.17, wspace=0.25)
    fig.savefig(f"../images/ch6/6.2/ch6_budget_rmse_bars{SUF}.png"); plt.close(fig)
print(f"wrote ../images/ch6/6.2/ch6_budget_rmse_bars{SUF}.png")
