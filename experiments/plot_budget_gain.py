"""Section 6.2 alternative main figure: paired reduction in test Spearman RMSE relative to each estimator's own 16 KiB
configuration, at 64 and 256 KiB. Per test seed: average over the four families of [RMSE(16 KiB config) - RMSE(b KiB config)],
configurations chosen on validation seeds; mean ± 2 SE over the 100 seeds. Source: results/budget_audit_cells.csv,
results/budget_seeds_ext.csv. Usage (from experiments/): python3 plot_budget_gain.py -> ../images/ch6/6.2/ch6_budget_accuracy_gain.png"""
import os
SUF = os.environ.get("BUDGET_SET", "")          # "" = n = 5000 (extended grid); "_n15k" = n = 15,000
SRC = SUF if SUF else "_ext"

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
C = pd.read_csv(f"results/budget_audit_cells{SUF}.csv")
SD = pd.read_csv(f"results/budget_seeds{SRC}.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
E = [("mpl_dskll", "RMPL", ce.COL["mle"]), ("xiao", "Counting Matrix", ce.COL["xiao"]), ("hermite", "Hermite Series", ce.COL["hermite"])]
SC = [("constant", "(a) Constant"), ("drift", "(b) Gradual drift"), ("jump", "(c) Abrupt jump")]


def per_seed(s, e, b):
    x = C[(C.scenario == s) & (C.estimator == e) & (C.budget_kib == b)]
    return 100 * np.mean([SD.loc[(r.family, s, e, r.approx, r.gain)].values for _, r in x.iterrows()], axis=0)


rows = []
for s, _ in SC:
    for e, _, _ in E:
        base = per_seed(s, e, 16)
        for b in (64, 256):
            d = base - per_seed(s, e, b)
            rows.append(dict(scenario=s, estimator=e, budget=b, red=d.mean(), se=d.std(ddof=1) / 10))
D = pd.DataFrame(rows)
bw = 0.27
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.3), sharey=True)
    for ax, (s, t) in zip(axs, SC):
        for i, (e, lab, col) in enumerate(E):
            ax.hlines(0, i - 1.5 * bw + 0.03, i - 0.5 * bw - 0.03, color=col, lw=2.2, alpha=0.35, zorder=5)   # 16 KiB: reference
            for j, b in enumerate((64, 256)):
                r = D[(D.scenario == s) & (D.estimator == e) & (D.budget == b)].iloc[0]
                x = i + j * bw
                ax.bar(x, r.red, width=bw * 0.92, color=col, alpha=0.6 if b == 64 else 1.0, zorder=3)
                ax.errorbar(x, r.red, yerr=2 * r.se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
        ax.axhline(0, color="#222222", lw=0.7, zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels(["RMPL", "Counting\nMatrix", "Hermite\nSeries"], fontsize=7); ax.tick_params(axis="x", length=0)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(t, fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("RMSE reduction from\n16 KiB ($\\times10^{-2}$)", fontsize=7.5)
    h = [plt.Line2D([], [], color="#777777", alpha=0.35, lw=2.2), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.6),
         plt.Rectangle((0, 0), 1, 1, color="#777777")]
    fig.legend(h, ["16 KiB (reference, 0)", "64 KiB", "256 KiB"], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
    fig.savefig(f"../images/ch6/6.2/ch6_budget_accuracy_gain{SUF}.png"); plt.close(fig)
print(D.round(3).to_string(index=False))
