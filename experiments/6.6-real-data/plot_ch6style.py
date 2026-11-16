"""Practical-application figures in the exact Chapter 6 style: plot_constant_ess.RC (copied verbatim from
bank-vff-study experiments/plot_constant_ess.py), grid #e4e3df, shared top-centre legend (frameless, fontsize 7, ncol 3),
printed width 6.1 in; strategy order and colours from the 6.3 conventions (Fixed #333333, VFF #8E6BB8, MFF #C2185B).
Data: loss_levels.csv (mean prequential log loss, x1e-3 nats/day/pair). No calculation here.
  ch6_practical_loss_bars  grouped bars from zero (no axis truncation), value labels, no whiskers on levels
  ch6_practical_loss       dot plot, every panel the same span in nats, value labels"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figures")
INK, GRID = "#0b0b0b", "#e4e3df"
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
      "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
      "axes.titlepad": 6, "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
      "figure.dpi": 300, "savefig.dpi": 300}
W = 6.1
L = pd.read_csv(os.path.join(HERE, "loss_levels.csv"))
M = [("fixed 399", r"Fixed ($N_{\mathrm{eff}}=399$)", "#333333", "s"), ("VFF", r"VFF ($\alpha=0.02$)", "#8E6BB8", "^"),
     ("MFF", "MFF", "#C2185B", "o")]
REG = [("all", "All days"), ("calm", "Calm\n(VIX < 20)"), ("stress", "Stress\n(VIX ≥ 25)")]
SETS = [("Gaussian", "Gaussian evaluation"), ("t4", r"$t_4$ evaluation")]
val = lambda s, m, r: L[(L.setting == s) & (L.method == m) & (L.regime == r)].loss_x1e3.iloc[0]
def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    fig.savefig(os.path.join(OUT, f"{name}_print.png"), dpi=110); plt.close(fig)

# ---------------------------------------------------------------- grouped bars (from zero)
with plt.rc_context(RC):
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    bw = 0.26; lo = min(val(s, m, r) for s, _ in SETS for m, *_ in M for r, _ in REG)
    for ax, (sk, sl) in zip(axs, SETS):
        for gi, (rk, rl) in enumerate(REG):
            for i, (m, lab, col, _) in enumerate(M):
                x = gi + (i - 1) * bw; y = val(sk, m, rk)
                ax.bar(x, y, width=bw * 0.92, color=col, edgecolor="white", lw=0.5, zorder=3)
                ax.text(x, y - 12, f"{y:.1f}", ha="center", va="top", fontsize=5.6, color="#52514e", rotation=90, zorder=4)
        ax.axhline(0, color="#8a8984", lw=0.6, zorder=2)
        ax.set_xticks(range(len(REG))); ax.set_xticklabels([l for _, l in REG], fontsize=7); ax.tick_params(axis="x", length=0)
        ax.set_title(sl); ax.set_ylim(lo * 1.22, 0)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    axs[0].set_ylabel("Mean log loss\n" + r"($\times10^{-3}$ nats per day per pair)")
    fig.legend([Patch(color=c) for _, _, c, _ in M], [l for _, l, _, _ in M], loc="upper center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.8, bottom=0.17, wspace=0.12)
    save(fig, "ch6_practical_loss_bars")

# ---------------------------------------------------------------- dot plot (common span)
span = max(max(val(s, m, r) for m, *_ in M) - min(val(s, m, r) for m, *_ in M) for s, _ in SETS for r, _ in REG) * 1.25
with plt.rc_context(RC):
    fig, axs = plt.subplots(3, 2, figsize=(W, 4.2), sharey=True)
    for ci, (sk, sl) in enumerate(SETS):
        for ri, (rk, rl) in enumerate(REG):
            ax = axs[ri, ci]; xs = [val(sk, m, rk) for m, *_ in M]; mid = (max(xs) + min(xs)) / 2
            for y, ((m, lab, col, mk), x) in enumerate(zip(M, xs)):
                ax.plot(x, y, marker=mk, color=col, ms=7.5 if mk == "^" else 5.5, mec="white", mew=0.6, ls="none", zorder=3)
                ax.annotate(f"{x:.1f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=6.3, color=col)
            ax.set_xlim(mid - span / 2, mid + span / 2); ax.set_ylim(len(M) - 0.5, -0.9)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10)); ax.set_yticks([])
            ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            if ri == 0: ax.set_title(sl)
            if ci == 0: ax.set_ylabel(rl.replace("\n", " "))
    fig.supxlabel(r"Mean log loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.012)
    fig.legend([Line2D([], [], color=c, marker=mk, ls="none", ms=7.5 if mk == "^" else 5.5, mec="white") for _, _, c, mk in M], [l for _, l, _, _ in M],
               loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.08, right=0.945, top=0.9, bottom=0.125, hspace=0.45, wspace=0.14)
    save(fig, "ch6_practical_loss")
print(f"dot-plot panel span {span:.1f} x1e-3 nats")
