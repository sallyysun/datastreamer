"""Clayton-Gumbel mixture regime (pi(Clayton) 0.75 -> 0.25 at tick 2500, rho_S constant): excess log loss by window,
shape weights, and chi_L / chi_U(0.05) paths, from the density2 run (ew_marginals.md §33).
Usage (repo root): python3 experiments/plot_mixture_regime.py -> notes/thesis/figures/matched/ch6_mixture_regime.png"""
import glob, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = "experiments/results/", "notes/thesis/figures/matched/"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)


d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(R + "family_bank_[0-9]*_density2_claygum_mix.csv")))
assert d.seed.nunique() == 60
path = (pd.concat(pd.read_csv(f) for f in sorted(glob.glob(R + "family_bank_path_[0-9]*_density2_claygum_mix.csv")))
        .groupby(["t", "series"]).value.mean().unstack())   # 4 chunks of 15 seeds, equal weight = 60-seed mean

fig = plt.figure(figsize=(11, 8.6))
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

# (a) excess log loss by window, mean +- 2 SE over 60 seeds
ax = fig.add_subplot(gs[0, 0])
style(ax, "(a) Excess log loss of each density forecast, by window")
wins = ["pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"]
wlab = ["before\n[500, 2500)", "0-100\nafter", "100-500\nafter", "500-1500\nafter", "after\n[2500, 5000)"]
meth = [("family_bank", "family-bank mixture", BLUE, "o"), ("mle_bank_log_point", "single Gaussian (MPL log bank)", ORANGE, "s"),
        ("mle_bank_log_mix", "Gaussian mixture (log-bank weights)", AQUA, "D"),
        ("arm_t_1999", "best fixed arm in hindsight (t4, M 1999)", VIOLET, "^")]
x = np.arange(len(wins))
for i, (m, lab, c, mk) in enumerate(meth):
    v = d[(d.method == m) & (d.metric == "excess_x1000")].pivot(index="seed", columns="window", values="value")[wins]
    mu, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
    ax.errorbar(x + (i - 1.5) * 0.14, mu, yerr=2 * se, fmt=mk, ms=6, color=c, lw=1.6, capsize=0, label=lab)
fl = d[(d.method == "gauss_floor_expected") & (d.metric == "excess_x1000")].groupby("window").value.mean()[wins]
ax.scatter(x, fl, marker="_", s=900, color=MUTED, lw=2, label="Gaussian floor D = KL(true || best Gaussian)", zorder=1)
ax.set_xticks(x); ax.set_xticklabels(wlab, fontsize=8)
ax.set_ylabel(r"excess log loss ($10^{-3}$ nats per observation)", fontsize=9, color=INK)
ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

# (b) shape weights
ax = fig.add_subplot(gs[0, 1])
style(ax, "(b) Family-bank weight on each shape (summed over memories)")
t = path.index.values
for s, lab, c in [("w_clayton", "Clayton", BLUE), ("w_t", "t4", VIOLET), ("w_gumbel", "Gumbel", ORANGE), ("w_gaussian", "Gaussian", AQUA)]:
    ax.plot(t, path[s], color=c, lw=2, label=lab)
    ax.annotate(lab, (5000, path[s].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
ax.set_xlim(500, 5000); ax.set_ylim(0, 1)
ax.set_ylabel("seed-mean weight", fontsize=9, color=INK)
ax.legend(fontsize=7.5, frameon=False, loc="center left")

# (c, d) tails
for j, (tail, lab) in enumerate([("chiL", r"(c) Lower tail $\chi_L(0.05)$"), ("chiU", r"(d) Upper tail $\chi_U(0.05)$")]):
    ax = fig.add_subplot(gs[1, j])
    style(ax, lab)
    ax.plot(t, path[tail + "_true"], color=INK, lw=2, label="truth (mixture)")
    ax.plot(t, path[tail + "_family"], color=BLUE, lw=2, label="family bank")
    ax.plot(t, path[tail + "_ratio399"], color=YELLOW, lw=2, label="EW corner counter, M = 399")
    ax.set_xlim(500, 5000); ax.set_ylim(0, 0.75)
    ax.set_xlabel("observation", fontsize=9, color=MUTED)
    ax.set_ylabel("seed-mean estimate", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
for a in fig.axes:
    if a is not fig.axes[0]:
        a.axvline(2500, color=MUTED, lw=0.9, ls=":")

fig.suptitle("Clayton-Gumbel mixture: Clayton weight 0.75 -> 0.25 at observation 2500, Spearman's rho fixed at 0.58 "
             "(truth not in the family bank)\n60 seeds (42-101); (a) mean ± 2 SE; (b)-(d) seed means",
             fontsize=10.5, color=INK, x=0.01, ha="left")
fig.savefig(OUT + "ch6_mixture_regime.png", dpi=150, bbox_inches="tight")
print("wrote " + OUT + "ch6_mixture_regime.png")
