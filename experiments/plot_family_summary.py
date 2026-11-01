"""One-page summary of the family bank against its fair competitors, one panel per question (seeds 42-101, Linux).
  (a) rho_S RMSE        family bank minus each Chapter 6 bank, per copula x scenario cell   results/family_bank_42*.csv
  (b) density           excess log loss at the true PIT, same comparison                     results/family_bank_42*.csv
  (c) tails chi(q)      family bank minus corner counters, Hermite and Xiao densities, per
                        regime x q x tail cell                                               results/tail_bank_*_42.csv
  (d) shape switches    ticks for the new family to reach 50% of the weight, per seed        results/family_bank_switch_42_*.csv
Each point: paired mean +- 2 SE over 60 seeds; blue = family bank better beyond 2 SE, orange = worse, grey = within.
Usage (from experiments/): python3 plot_family_summary.py -> ../notes/thesis/figures/ch6_family_summary.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
sfx = lambda c: "" if c == "gaussian" else "_" + c
fb = pd.concat(pd.read_csv(f"results/family_bank_42{sfx(c)}.csv") for c in
               ["gaussian", "clayton", "gumbel", "t", "gauss2t", "clayton2gumbel", "gauss2clayton"])
tb = pd.concat(pd.read_csv(f"results/tail_bank_{r}_42.csv") for r in
               ["gauss2clayton", "clayton2gumbel", "gauss2t", "mix", "gaussian"])


def strip(ax, wide, refs, title, xlabel):
    """wide: index = (cell..., seed), columns = methods, values already scaled; family_bank vs each ref."""
    cells = wide.index.droplevel(-1).unique()
    for j, (ref, lab) in enumerate(refs):
        better = 0
        for i, cell in enumerate(cells):
            x = (wide.loc[cell, "family_bank"] - wide.loc[cell, ref]).dropna()
            m, h = x.mean(), 2 * x.std(ddof=1) / len(x) ** .5
            col = BLUE if m + h < 0 else ORANGE if m - h > 0 else INK2
            better += m + h < 0
            y = -j + (i - (len(cells) - 1) / 2) * 0.5 / len(cells)
            ax.errorbar(m, y, xerr=h, fmt="o", ms=3, color=col, ecolor=col, elinewidth=0.8, capsize=0)
        ax.text(1.03, -j, f"{better}/{len(cells)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-300, 150)
    ax.set_yticks([-j for j in range(len(refs))], [lab for _, lab in refs])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.set_xlabel(xlabel)


banks = [("xiao_bank_log", "Xiao bank, Gaussian log loss"), ("hermite_bank_log", "Hermite bank, Gaussian log loss"),
         ("mle_bank_log", "MPL bank, Gaussian log loss"), ("xiao_bank_own", "Xiao bank, own density"),
         ("hermite_bank_own", "Hermite bank, own density"), ("mle_bank_own", "MPL bank, own density")]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
w = fb[(fb.window == "whole") & (fb.metric == "rmse")].pivot_table(index=["copula", "scenario", "seed"],
                                                                     columns="method", values="value") * 1000
strip(ax[0, 0], w, banks, "(a) Tracking $\\rho_S$: 19 copula x scenario cells",
      "family bank minus competitor, RMSE ($\\times10^{-3}$, symlog)")
w = fb[(fb.window == "whole") & (fb.metric == "excess_x1000")].pivot_table(index=["copula", "scenario", "seed"],
                                                                             columns="method", values="value")
strip(ax[0, 1], w, banks, "(b) Predictive density: 19 cells",
      "family bank minus competitor, excess log loss ($\\times10^{-3}$ nats/tick, symlog)")
tails = [("tail_chi_399", "corner counter, M = 399"), ("tail_chi_1999", "corner counter, M = 1999"),
         ("tail_chi_4999", "corner counter, M = 4999"), ("hermite_bank_own", "Hermite bank, own density"),
         ("hermite_1999", "Hermite density, M = 1999"), ("xiao_1999", "Xiao 20x20 histogram, M = 1999")]
w = tb[tb.window == "whole"].pivot_table(index=["regime", "q", "tail", "seed"], columns="method", values="rmse") * 1000
strip(ax[1, 0], w, tails, "(c) Tail dependence $\\chi(q)$, q = 0.05, 0.10: 20 regime x q x tail cells",
      "family bank minus competitor, RMSE ($\\times10^{-3}$, symlog)")

a = ax[1, 1]
sw = [("gauss2t", "Gaussian → $t_4$"), ("clayton2gumbel", "Clayton → Gumbel"), ("gauss2clayton", "Gaussian → Clayton")]
rng = np.random.default_rng(0)
for j, (c, lab) in enumerate(sw):
    s = pd.read_csv(f"results/family_bank_switch_42_{c}.csv")
    t = s.t50[s.t50 >= 0]
    a.scatter(t, -j + rng.uniform(-0.15, 0.15, len(t)), s=9, color=BLUE, alpha=0.6, lw=0)
    a.plot([t.median()] * 2, [-j - 0.25, -j + 0.25], color=INK, lw=1.6)
    a.text(t.median(), -j + 0.3, f"median {t.median():.0f}", ha="center", fontsize=7.5, color=INK)
null = pd.read_csv("results/family_bank_switch_42.csv")
n50 = null.groupby("seed").t50.apply(lambda x: (x >= 0).any()).sum()
a.set_yticks([-j for j in range(len(sw))], [l for _, l in sw])
a.set_xlim(0, 600)
a.grid(axis="x", color=GRID, lw=0.6)
a.set_title("(d) Shape switch at constant $\\rho_S$: ticks until\nthe new family holds 50% of the weight",
            color=INK, fontsize=9, loc="left")
a.set_xlabel(f"ticks after the switch (one dot per seed)\nno switch (Gaussian path): a wrong family reached 50% in {n50}/60 seeds")
fig.suptitle("Family bank against fair competitors. Points: one cell, paired mean ± 2 SE over 60 seeds; "
             "blue = family bank better beyond 2 SE, orange = worse, grey = within; n/N = cells better",
             fontsize=8.5, color=INK)
fig.tight_layout(rect=(0, 0, 0.97, 0.96), w_pad=4)
fig.savefig("../notes/thesis/figures/ch6_family_summary.png", dpi=170)
print("wrote ch6_family_summary.png")
