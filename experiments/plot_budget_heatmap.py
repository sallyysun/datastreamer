"""Section 6.2: RMPL minus the better nonparametric estimator (Counting Matrix or Hermite Series, the one with lower validation
RMSE at its validated configuration), test RMSE x1e-2, paired mean ± SE over 100 test seeds, at 64 KiB. Verdicts are identical
at 16 and 256 KiB (results/budget_audit_rmpl_vs_best.csv, from audit_budget.py).
Usage (from experiments/): python3 plot_budget_heatmap.py -> ../images/ch6/6.2/ch6_budget_rmpl_vs_np_heatmap.png"""
import os
SUF = os.environ.get("BUDGET_SET", "")          # "" = n = 5000 (extended grid); "_n15k" = n = 15,000
SRC = SUF if SUF else "_ext"

import sys
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
K = pd.read_csv(f"results/budget_audit_rmpl_vs_best{SUF}.csv")
K = K[K.budget_kib == 64].set_index(["scenario", "family"])
FAMS = [("gaussian", "Gaussian"), ("clayton", "Clayton"), ("gumbel", "Gumbel"), ("t", "$t_4$")]
SC = [("constant", "Constant"), ("drift", "Drift"), ("jump", "Jump")]
SH = {"xiao": "CM", "hermite": "HS"}
D = np.array([[K.loc[(s, f)].val_d for f, _ in FAMS] for s, _ in SC])
cmap = LinearSegmentedColormap.from_list("rmpl_np", ["#0f8a5f", "#ffffff", "#5b6b80"])
lim = 0.7
with plt.rc_context(ce.RC):
    fig, ax = plt.subplots(figsize=(core.W * 0.72, 2.2))
    im = ax.imshow(D, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, (s, _) in enumerate(SC):
        for j, (f, _) in enumerate(FAMS):
            r = K.loc[(s, f)]
            tie = r.val_v == "T"
            col = "white" if abs(r.val_d) > 0.42 else "#111111"
            ax.text(j, i - 0.12, f"{r.val_d:+.2f}".replace("-", "−") + ("" if not tie else "†"), ha="center", va="center", fontsize=7.5, color=col)
            ax.text(j, i + 0.22, f"± {r.val_se:.2f}  vs {SH[r.val_comp]}", ha="center", va="center", fontsize=5.8, color=col)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in FAMS], fontsize=7.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in SC], fontsize=7.5)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True); ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[-0.6, -0.3, 0, 0.3, 0.6])
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_visible(False)
    cb.set_label("$\\Delta$RMSE ($\\times10^{-2}$)", fontsize=7)
    fig.text(0.47, 0.03, "Green: RMPL better.  Grey: nonparametric better.  †: within 2 SE.", ha="center", fontsize=6.5, color="#333333")
    fig.subplots_adjust(left=0.13, right=0.88, top=0.85, bottom=0.14)
    fig.savefig(f"../images/ch6/6.2/ch6_budget_rmpl_vs_np_heatmap{SUF}.png"); plt.close(fig)
print(K[["val_comp", "val_d", "val_se", "val_v"]].round(3))
print(f"wrote ../images/ch6/6.2/ch6_budget_rmpl_vs_np_heatmap{SUF}.png")
