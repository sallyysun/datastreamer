"""Whole-copula accuracy of the three estimators' own-density banks (Xiao 5x5 histogram, Hermite series,
Gaussian MPL), from results/ise*.csv (probe_ise; 60 seeds). Left: density ISE on the normal-score
scale; right: copula CDF ISE. Same rows in both. See budget_mpl.md §18.
Usage (from experiments/): python3 plot_ise_estimators.py -> ../notes/thesis/figures/ch6_ise_estimators.png"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.concat(pd.read_csv(f) for f in glob.glob("results/ise*.csv"))
d = d[d.window == "whole"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST = [("xiao_bank_own", "Xiao (5×5 histogram)", "#2a78d6", "o"),
       ("hermite_bank_own", "Hermite (series)", "#eb6834", "s"),
       ("mle_bank_own", "MPL (Gaussian)", "#1baf7a", "^")]
NAME = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t₄",
        "clayton2gumbel": "Clayton → Gumbel", "gauss2t": "Gaussian → t₄"}
rows = [(f, s) for f in ["gaussian", "clayton", "gumbel", "t"] for s in ["constant", "drift", "jump"]] + \
       [("clayton2gumbel", "shape"), ("gauss2t", "shape")]
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
fig.patch.set_facecolor(SURF)
for a, (metric, scale, title) in zip(axes, [("ise_density", 1e3, r"(a) density ISE, normal-score scale ($\times10^{-3}$, log)"),
                                             ("ise_cdf", 1e4, r"(b) copula CDF ISE ($\times10^{-4}$, log)")]):
    for i, (f, s) in enumerate(rows):
        vals = []
        for k, (m, lab, col, mk) in enumerate(EST):
            x = d[(d.copula == f) & (d.scenario == s) & (d.method == m)][metric] * scale
            mu, h = x.mean(), 2 * x.std(ddof=1) / np.sqrt(len(x))
            vals.append(mu)
            a.errorbar(mu, i, xerr=h, fmt=mk, color=col, ms=7, capsize=0, lw=1.2, mec=SURF, mew=0.8,
                       label=lab if i == 0 else None, zorder=3)
        a.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2.5, zorder=1)
    a.set_xscale("log")
    a.set_title(title, fontsize=10.5, color=INK, loc="left")
    a.set_facecolor(SURF)
    a.grid(True, axis="x", color=GRID, lw=0.6)
    a.tick_params(labelsize=8.5, colors=MUTED)
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        a.spines[sp].set_color(GRID)
    for y in [2.5, 5.5, 8.5, 11.5]:
        a.axhline(y, color=GRID, lw=0.8, ls=":")
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([NAME[f] if s == "shape" else f"{NAME[f]} · {s}" for f, s in rows], fontsize=9, color=INK)
axes[0].set_ylim(len(rows) - 0.5, -0.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 0.985))
fig.suptitle("Whole-copula accuracy of the three estimators (own-density banks)", fontsize=12, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.915, "60 seeds, mean ± 2 SE; lower is better. Each method's own predictive density, mixed by its bank's weights.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("../notes/thesis/figures/ch6_ise_estimators.png", dpi=150)
print("wrote ../notes/thesis/figures/ch6_ise_estimators.png")
