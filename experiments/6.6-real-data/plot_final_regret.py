"""Regret figure (t4 copula): static (m = 0) and one-switch (m = 1) regret of fixed ESS 39 / 199 / 399, VFF and MFF,
relative to each pair's hindsight-best fixed ESS (m = 0) or best sequence of fixed arms with at most one switch (m = 1).
Supersedes Figure 2 of plot_final.py (VFF and MFF only). From saved outputs; no estimator is rerun.
  VFF, MFF   : switching_per_pair.csv (analyze_switching.py).
  fixed ESS k: regret_k = mean loss_k - oracle_m, with oracle_m = mean loss_MFF - regret_MFF,m from the same file and the
               same evaluation days (t >= 560, 2008 onward). Checked against a direct static computation for m = 0.
Bars beyond the axis limit are cut at the edge and labelled with their value.
Writes final_regret_values.csv and images/ch6_practical_final_regret.png. Usage: python3 plot_final_regret.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "RMPL", "practical_rmpl"), ("Counting matrix", "Counting matrix", "practical_xiao"), ("Hermite series", "Hermite", "practical_hermite")]
FIX = [39, 99, 199, 399, 1999]; SHOW_FIX = FIX
# Chapter 6 all-fixed bar style (experiments/plot_strategy_allfixed.py on bank-vff-study), turned horizontal
C_OTHER, C_BEST, C_VFF, C_MFF = "#cfcfcf", "#333333", "#8E6BB8", "#C2185B"
PLOT_FIX = [99, 199, 399]   # ESS 39 and 1999 stay in the benchmark (an MFF candidate) but is not drawn
BARS = [f"fixed{k}" for k in PLOT_FIX] + ["VFF", "MFF"]
Sw = pd.read_csv(os.path.join(HERE, "results", "switching_per_pair.csv")); Sw = Sw[Sw.copula == "t4"]
rows, worst = [], 0.0
for lab, swlab, d in EST:
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        p = os.path.basename(f)[:-4]
        x = pd.read_csv(f, parse_dates=["date"]).iloc[560:]; x = x[x.date >= "2008-01-01"]
        L = {k: -1e3 * x[f"lt_fixed_{k}"].mean() for k in FIX}; Lm = -1e3 * x.lt_mffmix_t.mean()
        for m in (0, 1):
            g = Sw[(Sw.estimator == swlab) & (Sw.pair == p) & (Sw.m == m)].set_index("method").regret
            orc = Lm - g["MFF"]
            if m == 0: worst = max(worst, abs(orc - min(L.values())))
            for k in SHOW_FIX: rows.append(dict(estimator=lab, m=m, method=f"fixed{k}", pair=p, regret=L[k] - orc))
            for k in ("VFF", "MFF"): rows.append(dict(estimator=lab, m=m, method=k, pair=p, regret=g[k]))
print(f"CHECK static oracle from switching_per_pair.csv vs min fixed loss: max |diff| = {worst:.2e}")
R = pd.DataFrame(rows).groupby(["estimator", "m", "method"]).regret.mean()
R.to_csv(os.path.join(HERE, "results", "final_regret_values.csv"))
XMAX = 7.5
with plt.rc_context(RC):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True, sharex=True)
    bw = 0.13
    for ax, (m, ml) in zip(axs, ((0, "Static regret ($m=0$)"), (1, "One-switch regret ($m=1$)"))):
        for ei, (e, _, _) in enumerate(EST):
            best = min((f"fixed{k}" for k in PLOT_FIX), key=lambda k: R[(e, m, k)])
            for i, k in enumerate(BARS):
                v = R[(e, m, k)]; y = ei + (i - (len(BARS) - 1) / 2) * bw
                col = C_VFF if k == "VFF" else C_MFF if k == "MFF" else C_BEST if k == best else C_OTHER
                ax.barh(y, min(v, XMAX), height=bw * 0.62, color=col, edgecolor="#555555", lw=0.6, zorder=3)   # outlined, as in the box plots
                if v > XMAX:   # cut bar: white slash and value beyond the edge
                    ax.plot([XMAX * 0.955, XMAX * 0.975], [y + bw * 0.3, y - bw * 0.3], color="white", lw=1.2, zorder=5)
                    ax.text(XMAX * 1.01, y, f"{v:.1f}", ha="left", va="center", fontsize=5.4, color="#52514e", clip_on=False)
                if k == best:
                    ax.text(max(v, 0) + 0.12, y, k[5:], ha="left", va="center", fontsize=5.4, color=C_BEST)
        ax.axvline(0, color="#52514e", lw=0.8, zorder=4)
        ax.set_xlim(-3.0, XMAX); ax.set_ylim(len(EST) - 0.5, -0.5); ax.set_title(ml)
        ax.set_yticks(range(len(EST))); ax.set_yticklabels([e for e, _, _ in EST]); ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    fig.supxlabel(r"Average regret ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.02)
    sq = lambda c: plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#555555", lw=0.6)
    fig.legend([sq(C_OTHER), sq(C_BEST), sq(C_VFF), sq(C_MFF)],
               [r"Fixed $N_{\mathrm{eff}}$ = 99, 199, 399 (top to bottom)", "Fixed, lowest average regret (labelled)",
                "VFF", "MFF"], loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0),
               handlelength=1.2, columnspacing=1.5)
    fig.subplots_adjust(left=0.19, right=0.965, top=0.8, bottom=0.15, wspace=0.12)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_final_regret.png"))
    plt.close(fig)
print(R.unstack("method")[BARS].round(2).to_string())
