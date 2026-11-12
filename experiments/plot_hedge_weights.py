"""Step 4 figure (chapter6_metrics.md §16): MPL log-bank weights around the jump (d 0.2 -> 0.8 at t = 2500) under plain Hedge
(delta = 0) and fixed share (delta = 1e-4), seed means over 60 seeds, and the weight-averaged memory sum_k w_k N_eff,k.
Reads results/thesis_bank_weights_{hedge0,density4}_s<first>[_fam].csv (probe_thesis_bank; jump scenario, every 5 ticks).
Usage (repo root): python3 experiments/plot_hedge_weights.py [main|all] -> notes/thesis/figures/matched/ch6_hedge_vs_fixedshare[_all].png"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = "experiments/results/", "notes/thesis/figures/matched/"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MEMCOL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # one blue ramp, short memory light -> long dark
HEDGE, FS = "#eb6834", "#2a78d6"
FAMLAB = {"gaussian": "Gaussian copula", "clayton": "Clayton copula", "gumbel": "Gumbel copula", "t": "t4 copula"}
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
fams = ["gaussian", "clayton"] if MODE == "main" else ["gaussian", "clayton", "gumbel", "t"]


def weights(fam, tag):
    s = "" if fam == "gaussian" else "_" + fam
    d = pd.concat([pd.read_csv(f"{R}thesis_bank_weights_{tag}_s{b}{s}.csv") for b in (42, 57, 72, 87)])
    d = d[d.bank == "bank_log"].groupby(["t", "mem"]).weight.mean().unstack()   # equal chunks of 15 seeds -> 60-seed mean
    return d


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)


fig, axes = plt.subplots(len(fams), 3, figsize=(12, 3.1 * len(fams) + 0.9), squeeze=False)
fig.patch.set_facecolor(SURF)
for i, fam in enumerate(fams):
    W = {"hedge0": weights(fam, "hedge0"), "density4": weights(fam, "density4")}
    for j, (tag, lab) in enumerate([("hedge0", "plain Hedge (δ = 0)"), ("density4", "fixed share (δ = 10⁻⁴)")]):
        ax = axes[i, j]
        style(ax, f"{lab}" if i == 0 else "")
        w = W[tag]
        ax.stackplot(w.index, *[w[m] for m in w.columns], colors=MEMCOL, linewidth=0.4, edgecolor=SURF,
                     labels=[f"N_eff {m}" for m in w.columns])
        ax.axvline(2500, color=INK, lw=0.9, ls=":")
        ax.set_xlim(2000, 4000); ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel(FAMLAB[fam] + "\n\nseed-mean weight", fontsize=9, color=INK)
        if i == len(fams) - 1:
            ax.set_xlabel("observation (jump at 2500)", fontsize=9, color=MUTED)
    ax = axes[i, 2]
    style(ax, "weight-averaged memory Σ w_k N_eff,k" if i == 0 else "")
    for tag, lab, c in [("hedge0", "plain Hedge", HEDGE), ("density4", "fixed share", FS)]:
        w = W[tag]
        mem = (w * np.array(w.columns, float)).sum(axis=1)
        ax.plot(mem.index, mem, color=c, lw=2, label=lab)
    ax.axvline(2500, color=INK, lw=0.9, ls=":")
    ax.set_yscale("log"); ax.set_xlim(2000, 4000); ax.set_ylim(30, 2500)
    ax.set_yticks([39, 99, 199, 399, 1999]); ax.set_yticklabels(["39", "99", "199", "399", "1999"])
    ax.minorticks_off()
    if i == len(fams) - 1:
        ax.set_xlabel("observation (jump at 2500)", fontsize=9, color=MUTED)
    if i == 0:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.36, -0.005))
fig.suptitle("MPL log bank around a jump in dependence (d 0.2 → 0.8): plain Hedge vs fixed share, same arms, η and streams\n"
             "seed means over 60 seeds (42–101); only δ differs", fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.6 / fig.get_figheight()))
out = OUT + ("ch6_hedge_vs_fixedshare.png" if MODE == "main" else "ch6_hedge_vs_fixedshare_all.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote " + out)
