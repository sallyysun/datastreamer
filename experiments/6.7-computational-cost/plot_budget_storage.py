"""Section 6.2: actual retained-state storage of the validation-selected configurations at each storage budget. Bars = maximum
over the 12 family x scenario selections (labelled in KiB). Storage = peak store_bytes() over every tick, max over validation seeds (all fit on test seeds too), in KiB
(1 KiB = 1024 B). RMPL counts both DataSketches KLL margins and its parameter / gain state. Source: results/budget_audit_cells.csv
(audit_budget.py). Usage: python3 plot_budget_storage.py -> images/ch6_budget_storage.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as cfg  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
C = pd.read_csv("results/budget_selected.csv")
C["kib"] = C.peak_val / 1024
E = [("mpl_dskll", "RMPL", ce.COL["mle"]), ("xiao", "Counting Matrix", ce.COL["xiao"]), ("hermite", "Hermite Series", ce.COL["hermite"])]
B = cfg.BUDGETS
bw = 0.26
with plt.rc_context(ce.RC):
    fig, ax = plt.subplots(figsize=(core.W * 0.72, 2.5))
    for i, b in enumerate(B):
        ax.hlines(b, i - 0.45, i + 0.45, color="#444444", lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.text(i - 0.45, b * 1.07, "budget", va="bottom", ha="left", fontsize=6, color="#444444")
        for j, (e, lab, col) in enumerate(E):
            x = C[(C.estimator == e) & (C.budget_kib == b)].kib
            xp = i + (j - 1) * bw
            ax.bar(xp, x.max(), width=bw * 0.9, color=col, alpha=0.85, zorder=3, label=lab if i == 0 else None)
            ax.text(xp, x.max() * 0.9, f"{x.max():.1f}", ha="center", va="top", fontsize=5.6, color="white")
    ax.set_yscale("log", base=2)
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256]); ax.set_yticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_ylim(2, 400); ax.set_xlim(-0.55, 2.75)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{b} KiB" for b in B]); ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Storage budget", fontsize=7.5); ax.set_ylabel("Maximum selected\nretained state (KiB)", fontsize=7.5)
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.53, 1.0), handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.17)
    fig.savefig("images/ch6_budget_storage_n15k.png"); plt.close(fig)
print(C.groupby(["estimator", "budget_kib"]).kib.agg(["min", "median", "max"]).round(2))
print("wrote images/ch6_budget_storage_n15k.png")
