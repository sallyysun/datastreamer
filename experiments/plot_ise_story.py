# Two Chapter 6 figures from existing probe_batch_ise output (no new runs): family bank vs the
# Gaussian-working-model MPL bank. See notes/studies/thesis_bank_ise_batch.md.
# Usage (repo root): python3 experiments/plot_ise_story.py
import glob
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FB = ("Family bank", "#2a78d6", "o")
MB = ("Gaussian MPL bank", "#eb6834", "s")
NAME = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t₄",
        "clayton2gumbel": "Clayton → Gumbel", "gauss2clayton": "Gaussian → Clayton"}

d = pd.concat(pd.read_csv(f) for f in glob.glob("experiments/results/batch_ise_42*.csv"))
d = d[(d.window == "whole") & d.method.isin(["family_bank", "mle_bank_log"])]
p = d.pivot_table(index=["copula", "scenario", "seed", "metric"], columns="method", values="value").reset_index()
g = p.groupby(["copula", "scenario", "metric"]).agg(
    fb=("family_bank", "mean"), mb=("mle_bank_log", "mean"),
    fse=("family_bank", lambda x: x.std() / np.sqrt(len(x))),
    mse=("mle_bank_log", lambda x: x.std() / np.sqrt(len(x)))).reset_index()


def label(c, s):
    return NAME[c] if s == "shape" else f"{NAME[c]} · {s}"


def get(c, s, m):
    return g[(g.copula == c) & (g.scenario == s) & (g.metric == m)].iloc[0]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def dumbbell(ax, rows, metric, scale, log):
    y = np.arange(len(rows))
    for i, (c, s) in enumerate(rows):
        r = get(c, s, metric)
        ax.plot([scale * r.fb, scale * r.mb], [i, i], color=GRID, lw=2.5, zorder=1)
        for v, se, (lab, col, mk) in [(r.fb, r.fse, FB), (r.mb, r.mse, MB)]:
            ax.errorbar(scale * v, i, xerr=2 * scale * se, fmt=mk, color=col, ms=7, capsize=0, lw=1.2,
                        zorder=2, label=lab if i == 0 else None, mec=SURF, mew=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label(c, s) for c, s in rows], fontsize=9, color=INK)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    if log:
        ax.set_xscale("log")
    style(ax)


def ratio_col(ax, rows, metric, x):
    for i, (c, s) in enumerate(rows):
        r = get(c, s, metric)
        ax.text(x, i, f"{r.mb / r.fb:.1f}×", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, color=INK)


# Figure 1: ISE in the scenarios where the two banks differ, plus one where the Gaussian model is right.
rows1 = [("clayton", "constant"), ("gumbel", "constant"), ("clayton2gumbel", "shape"),
         ("gauss2clayton", "shape"), ("clayton", "drift"), ("clayton", "jump"), ("gaussian", "drift")]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
fig.patch.set_facecolor(SURF)
dumbbell(ax, rows1, "ise_x1e4", 1.0, log=True)
ax.axhline(len(rows1) - 1.5, color=MUTED, lw=0.8, ls=":")
ax.text(1.02, len(rows1) - 1, "Gaussian model\ncorrect", transform=ax.get_yaxis_transform(), fontsize=7.5,
        color=MUTED, va="center", ha="left")
ratio_col(ax, rows1[:-1], "ise_x1e4", 1.02)
ax.text(1.02, -1.0, "MPL ÷ family", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, ha="left")
ax.set_xlabel(r"copula ISE $\int(\hat C-C)^2$ over the run  ($\times10^{-4}$, log scale; lower is better)",
              fontsize=9, color=MUTED)
ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
fig.suptitle("Under misspecification the family bank recovers the copula far more accurately",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.905, "Mean ± 2 SE over 60 seeds. Each row is one scenario.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 0.9, 0.9))
fig.savefig("notes/thesis/figures/ch6_ise_paired.png", dpi=150)

# Figure 2: the same rows, two metrics side by side; sorted by the ISE ratio.
rows2 = [(c, s) for c in ["gaussian", "clayton", "gumbel", "t"] for s in ["constant", "drift", "jump"]]
rows2 += [("clayton2gumbel", "shape"), ("gauss2clayton", "shape")]
rows2.sort(key=lambda cs: -get(*cs, "ise_x1e4").mb / get(*cs, "ise_x1e4").fb)
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 6.2), sharey=True)
fig.patch.set_facecolor(SURF)
dumbbell(a, rows2, "rmse", 1e3, log=False)
dumbbell(b, rows2, "ise_x1e4", 1.0, log=True)
a.set_title(r"(a) Tracking $\rho_S$: RMSE ($\times10^{-3}$)", fontsize=10, color=INK, loc="left")
b.set_title(r"(b) Recovering the copula: ISE ($\times10^{-4}$, log)", fontsize=10, color=INK, loc="left")
ratio_col(a, rows2, "rmse", 1.01)
ratio_col(b, rows2, "ise_x1e4", 1.01)
for ax in (a, b):
    ax.text(1.01, -1.1, "MPL ÷\nfamily", transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED,
            ha="left", va="bottom")
a.set_xlim(left=15)
fig.legend(*a.get_legend_handles_labels(), fontsize=9, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.985))
fig.suptitle(r"Similar $\rho_S$ error, very different copula error",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.925, "Same scenario rows in both panels, sorted by the ISE ratio. Mean ± 2 SE over 60 seeds; "
         "oscillate and Gaussian → t₄ are discussed separately.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=4)
fig.savefig("notes/thesis/figures/ch6_ise_two_metrics.png", dpi=150)
print("wrote notes/thesis/figures/ch6_ise_paired.png, ch6_ise_two_metrics.png")
