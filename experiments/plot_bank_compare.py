"""Xiao vs Hermite vs MPL log banks (regenerated Chapter 6 data: results/thesis_bank[_fam].csv, 60 seeds,
spec MPL arms with the start-up rule). Left: whole-run RMSE, drift and jump, all four families.
Right: RMSE by window around the jump. Paired differences vs the MPL bank are in docs/chapter6_results.md.
Usage (from experiments/): python3 plot_bank_compare.py -> ../notes/thesis/figures/ch6_bank_compare.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.concat(pd.read_csv(f"results/thesis_bank{s}.csv") for s in ["", "_clayton", "_gumbel", "_t"])
d = d[d.method == "bank_log"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST = [("xiao", "Xiao bank", "#2a78d6", "o"), ("hermite", "Hermite bank", "#eb6834", "s"), ("mle", "MPL bank", "#1baf7a", "^")]
FAM = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t₄"}


def stat(x):
    return 1e3 * x.mean(), 2e3 * x.std(ddof=1) / np.sqrt(len(x))


def style(a):
    a.set_facecolor(SURF)
    a.grid(True, axis="x", color=GRID, lw=0.6)
    a.tick_params(labelsize=8.5, colors=MUTED)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        a.spines[s].set_color(GRID)


fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor(SURF)

# (a) whole-run RMSE, rows = scenario x family
rows = [(sc, f) for sc in ["drift", "jump"] for f in FAM]
for i, (sc, f) in enumerate(rows):
    for k, (c, lab, col, mk) in enumerate(EST):
        m, h = stat(d[(d.copula == f) & (d.scenario == sc) & (d.window == "whole") & (d.component == c)].rmse)
        a.errorbar(m, i + (k - 1) * 0.22, xerr=h, fmt=mk, color=col, ms=6.5, capsize=0, lw=1.2,
                   mec=SURF, mew=0.8, label=lab if i == 0 else None)
a.set_yticks(range(len(rows)))
a.set_yticklabels([f"{FAM[f]} · {sc}" for sc, f in rows], fontsize=9, color=INK)
a.set_ylim(len(rows) - 0.5, -0.5)
a.axhline(3.5, color=MUTED, lw=0.8, ls=":")
a.set_xlabel(r"whole-run RMSE of $\hat\rho_S$ ($\times10^{-3}$, lower is better)", fontsize=9, color=MUTED)
a.set_title("(a) Whole run, by family and scenario", fontsize=10.5, color=INK, loc="left")
style(a)

# (b) jump, by window: paired difference from the MPL bank (same seed, family), pooled over the four families
wins = [("pre", "before\n[500, 2500)"), ("rec_0_100", "0–100\nafter"), ("rec_100_500", "100–500\nafter"),
        ("rec_500_1500", "500–1500\nafter"), ("whole", "whole\nrun")]
x = np.arange(len(wins))
jp = d[d.scenario == "jump"].pivot_table(index=["copula", "seed", "window"], columns="component", values="rmse").reset_index()
for k, (c, lab, col, mk) in enumerate(EST[:2]):
    ms, hs = [], []
    for w, _ in wins:
        g = jp[jp.window == w]
        m, h = stat(g[c] - g["mle"])
        ms.append(m); hs.append(h)
    b.errorbar(x + (k - 0.5) * 0.18, ms, yerr=hs, fmt=mk, color=col, ms=7, lw=1.5, capsize=0, mec=SURF, mew=0.8)
b.axhline(0, color=EST[2][2], lw=1.6, label="MPL bank (reference)")
b.set_xticks(x)
b.set_xticklabels([l for _, l in wins], fontsize=8.5, color=INK)
b.set_ylabel(r"RMSE minus MPL bank ($\times10^{-3}$, paired)" + "\n> 0: MPL bank better", fontsize=9, color=MUTED)
b.set_title("(b) Jump, by window: difference from the MPL bank", fontsize=10.5, color=INK, loc="left")
b.set_facecolor(SURF)
b.grid(True, axis="y", color=GRID, lw=0.6)
b.tick_params(labelsize=8.5, colors=MUTED)
for s_ in ["top", "right"]:
    b.spines[s_].set_visible(False)
for s_ in ["left", "bottom"]:
    b.spines[s_].set_color(GRID)

fig.legend(*a.get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 0.985))
fig.suptitle("Log-loss banks: Xiao vs Hermite vs MPL arms", fontsize=12, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.905, "60 seeds × 4 families. (a) mean ± 2 SE; (b) paired over seed and family, ± 2 SE.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig("../notes/thesis/figures/ch6_bank_compare.png", dpi=150)
print("wrote ../notes/thesis/figures/ch6_bank_compare.png")
