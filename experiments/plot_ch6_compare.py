"""Chapter 6 comparison figures, from results already on disk (seeds 42-101, Linux).
  ch6_bank_forest.png    log bank minus each competitor, rho_S RMSE (mean +- 2 SE, paired over seeds), per family
                         x estimator, drift and jump. results/thesis_bank[_fam].csv (probe_thesis_bank).
  ch6_bank_recovery.png  rho_S RMSE by window after the jump, averaged over the 4 families, per estimator.
  ch6_tail_chi.png       seed-mean chi_L(0.05) around each switch: family bank, tail_chi M = 1999 / 4999, truth.
                         results/tail_bank_path_<regime>_42.csv (probe_tail_bank).
Usage (from experiments/): python3 plot_ch6_compare.py -> ../notes/thesis/figures/"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "../notes/thesis/figures"
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

fams = ["gaussian", "clayton", "gumbel", "t"]
FAM = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "$t_4$"}
COMP = {"xiao": "Xiao", "hermite": "Hermite", "mle": "MPL"}
d = pd.concat(pd.read_csv(f"results/thesis_bank{'' if f == 'gaussian' else '_' + f}.csv") for f in fams)

# 1. Forest plot -------------------------------------------------------------------------------------
refs = [("fixed_199", "fixed M = 199"), ("best", "best fixed M (hindsight)"), ("vff_slow", "VFF slow"),
        ("vff_fast", "VFF fast"), ("uniform", "uniform weights"), ("bank", "squared bank")]
w = d[d.window == "whole"].pivot_table(index=["copula", "scenario", "component", "seed"], columns="method",
                                       values="rmse") * 1000
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
ECOL = {"xiao": BLUE, "hermite": ORANGE, "mle": AQUA}
FMK = {"gaussian": "o", "clayton": "s", "gumbel": "^", "t": "D"}
for ax, sc in zip(axes, ["drift", "jump"]):
    for j, (ref, lab) in enumerate(refs):
        n_better = 0
        for fi, f in enumerate(fams):
            for ci, c in enumerate(COMP):
                g = w.loc[(f, sc, c)]
                other = g[g[[x for x in g.columns if x.startswith("fixed_")]].mean().idxmin()] if ref == "best" else g[ref]
                x = g.bank_log - other
                m, h = x.mean(), 2 * x.std(ddof=1) / len(x) ** .5
                n_better += m + h < 0
                y = -j + (fi * 3 + ci - 5.5) * 0.06
                ax.errorbar(m, y, xerr=h, fmt=FMK[f], ms=4, color=ECOL[c], ecolor=ECOL[c], elinewidth=0.9,
                            mec="white", mew=0.5, capsize=0)
        ax.text(1.0, -j, f"{n_better}/12", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(f"{sc}", color=INK)
    ax.set_xlabel(r"log bank minus competitor, RMSE of $\rho_S$ ($\times10^{-3}$, symlog)")
axes[0].set_yticks([-j for j in range(len(refs))], [lab for _, lab in refs])
from matplotlib.lines import Line2D
h1 = [Line2D([], [], color=ECOL[c], marker="o", ls="", label=COMP[c]) for c in COMP]
h2 = [Line2D([], [], color=INK2, marker=FMK[f], ls="", label=FAM[f]) for f in fams]
fig.legend(handles=h1 + h2, loc="lower center", ncol=7, frameon=False, fontsize=8)
fig.suptitle("Log bank against each competitor: each point is one family x estimator cell (mean ± 2 SE, "
             "60 seeds); left of 0 = log bank better; n/12 = cells better beyond 2 SE", fontsize=8.5, color=INK)
fig.tight_layout(rect=(0, 0.06, 0.97, 0.95))
fig.savefig(f"{OUT}/ch6_bank_forest.png", dpi=170)
plt.close(fig)

# 2. Recovery after the jump ---------------------------------------------------------------------------
wins = [("pre", "before\n[500, 2500)"), ("rec_0_100", "0-100\nafter"), ("rec_100_500", "100-500\nafter"),
        ("rec_500_1500", "500-1500\nafter")]
meth = [("fixed_39", "M = 39"), ("fixed_199", "M = 199"), ("fixed_1999", "M = 1999"), ("uniform", "uniform"),
        ("vff_fast", "VFF fast"), ("bank", "squared bank"), ("bank_log", "log bank")]
style = {"fixed_39": (GRID, None), "fixed_199": ("#c3c2b7", None), "fixed_1999": (INK2, None),
         "uniform": ("#c3c2b7", "//"), "vff_fast": (AQUA, None), "bank": (ORANGE, None), "bank_log": (BLUE, None)}
r = d[d.scenario == "jump"].groupby(["component", "window", "method"]).rmse.mean() * 1000
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
bw = 0.8 / len(meth)
for ax, c in zip(axes, COMP):
    for k, (m, lab) in enumerate(meth):
        vals = [r[(c, wn, m)] for wn, _ in wins]
        col, hatch = style[m]
        ax.bar(np.arange(len(wins)) + (k - len(meth) / 2 + .5) * bw, vals, bw * 0.9, color=col, hatch=hatch,
               edgecolor="white" if hatch is None else INK2, lw=0.4, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(range(len(wins)), [l for _, l in wins], fontsize=7.5)
    ax.set_title(COMP[c], color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"RMSE of $\rho_S$ ($\times10^{-3}$, log)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=7, fontsize=7.5, frameon=False)
fig.suptitle("Jump 0.2 → 0.8 at tick 2500: RMSE by window, mean over 4 families × 60 seeds", fontsize=9, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{OUT}/ch6_bank_recovery.png", dpi=170)
plt.close(fig)

# 3. chi(q) around the switch ---------------------------------------------------------------------------
regs = [("gauss2clayton", "Gaussian → Clayton"), ("clayton2gumbel", "Clayton → Gumbel"),
        ("gauss2t", "Gaussian → $t_4$"), ("mix", "Clayton/Gumbel mixture, π 0.75 → 0.25")]
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
for ax, (rg, title) in zip(axes.flat, regs):
    p = pd.read_csv(f"results/tail_bank_path_{rg}_42.csv")
    p = p[(p.t >= 3000) & (p.t < 9000)]
    s = lambda k: p[p.series == f"chiL_q05_{k}"].set_index("t").value
    ax.plot(s("true").index, s("true"), color=INK, lw=1.2, ls="--", label="true $\\chi_L(0.05)$")
    ax.plot(s("tail_chi_4999").index, s("tail_chi_4999"), color=AQUA, lw=1.6, label="corner counter, M = 4999")
    ax.plot(s("tail_chi_1999").index, s("tail_chi_1999"), color=ORANGE, lw=1.6, label="corner counter, M = 1999")
    ax.plot(s("family_bank").index, s("family_bank"), color=BLUE, lw=2, label="family bank")
    ax.axvline(5000, color=INK2, lw=0.6, ls=":")
    ax.set_title(title, color=INK, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
for ax in axes[:, 0]: ax.set_ylabel(r"$\chi_L(0.05)$, mean of 60 seeds")
for ax in axes[1]: ax.set_xlabel("tick (switch at 5000)")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/ch6_tail_chi.png", dpi=170)
print("wrote ch6_bank_forest.png, ch6_bank_recovery.png, ch6_tail_chi.png")
