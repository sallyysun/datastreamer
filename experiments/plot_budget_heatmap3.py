"""Section 6.2: three-way comparison at 64 KiB. For each family x scenario cell, each estimator's test RMSE minus the lowest of
the three (paired per test seed; configurations chosen on validation seeds). 0 marks the most accurate estimator in the cell;
'≈' marks an estimator within 2 SE (paired) of the best. BUDGET_SET=_n15k uses the n = 15,000 run.
Usage (from experiments/): python3 plot_budget_heatmap3.py -> ../images/ch6/6.2/ch6_budget_heatmap3way[_n15k].png"""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SUF = os.environ.get("BUDGET_SET", "")
SRC = SUF if SUF else "_ext"
sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
C = pd.read_csv(f"results/budget_audit_cells{SUF}.csv")
C = C[C.budget_kib == 64]
SD = pd.read_csv(f"results/budget_seeds{SRC}.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
FAMS = [("gaussian", "Gauss."), ("clayton", "Clayton"), ("gumbel", "Gumbel"), ("t", "$t_4$")]
SC = [("constant", "Constant"), ("drift", "Drift"), ("jump", "Jump")]
E = [("mpl_dskll", "(a) RMPL"), ("xiao", "(b) Counting Matrix"), ("hermite", "(c) Hermite Series")]
D = {e: np.zeros((3, 4)) for e, _ in E}
NEAR = {e: np.zeros((3, 4), bool) for e, _ in E}
rows = []
for i, (s, _) in enumerate(SC):
    for j, (f, _) in enumerate(FAMS):
        x = {}
        for e, _ in E:
            r = C[(C.family == f) & (C.scenario == s) & (C.estimator == e)].iloc[0]
            x[e] = 100 * SD.loc[(f, s, e, int(r.approx), float(r.gain))].values
        best = min(x, key=lambda e: x[e].mean())
        for e, _ in E:
            d = x[e] - x[best]
            D[e][i, j] = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d)) if e != best else 0.0
            NEAR[e][i, j] = e != best and abs(d.mean()) <= 2 * se
            rows.append(dict(scenario=s, family=f, estimator=e, excess=d.mean(), se=se, best=best == e))
pd.DataFrame(rows).to_csv(f"results/budget_heatmap3way{SUF}.csv", index=False)
cmap = LinearSegmentedColormap.from_list("excess", ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])   # green = best
vmax = max(D[e].max() for e, _ in E)
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.05))
    for ax, (e, t) in zip(axs, E):
        im = ax.imshow(D[e], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(3):
            for j in range(4):
                v = D[e][i, j]
                lab = "best" if v == 0 else (("<0.01" if v < 0.005 else f"{v:.2f}") + ("≈" if NEAR[e][i, j] else ""))
                ax.text(j, i, lab, ha="center", va="center", fontsize=6.6, color="white" if (v == 0 or v > 0.75 * vmax) else "#111111",
                        fontweight="bold" if v == 0 else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in FAMS], fontsize=6.2); ax.xaxis.tick_top()
        ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in SC] if e == "mpl_dskll" else [], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4), minor=True); ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
        ax.set_title(t, fontsize=8, pad=14, loc="left")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    cb.set_label("RMSE above the best ($\\times10^{-2}$)", fontsize=6.5)
    fig.text(0.45, 0.02, "best: most accurate in the cell.  ≈: within 2 SE of the best (paired).", ha="center", fontsize=6.3, color="#333333")
    fig.subplots_adjust(left=0.08, right=0.89, top=0.78, bottom=0.1, wspace=0.12)
    fig.savefig(f"../images/ch6/6.2/ch6_budget_heatmap3way{SUF}.png"); plt.close(fig)
print(f"wrote ../images/ch6/6.2/ch6_budget_heatmap3way{SUF}.png")
