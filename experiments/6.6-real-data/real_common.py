"""Shared constants, loaders and plot style for the Section 6.6 real-data scripts."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = [39, 99, 199, 399, 1999]
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
ECOL = {"RMPL": "#1baf7a", "Counting matrix": "#2a78d6", "Hermite series": "#eb6834"}
EVAL_ROW, EVAL_FROM = 560, "2008-01-01"
INK, GRID, W = "#0b0b0b", "#e4e3df", 6.1
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
      "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
      "axes.titlepad": 6, "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
      "figure.dpi": 300, "savefig.dpi": 300}


def result(name):
    return os.path.join(HERE, "results", name)


def image(name):
    return os.path.join(HERE, "images", name)


def outputs(S, d):
    """{pair: path} of the probe outputs in S/d/out, sorted by pair."""
    return {os.path.basename(f)[:-4]: f for f in sorted(glob.glob(f"{S}/{d}/out/*.csv"))}


def read(f):
    return pd.read_csv(f, parse_dates=["date"])


def eval_days(x):
    """Evaluation days: from row 560 (the warm-up) and from 2008 onward."""
    x = x.iloc[EVAL_ROW:]
    return x[x.date >= EVAL_FROM]


def change(x, w=99, lag=21):
    """|rho_roll_w[t] - rho_roll_w[t - lag]|, both held before the day, in the pair's own trading days."""
    return (x[f"rho_roll_{w}"] - x[f"rho_roll_{w}"].shift(lag)).abs()


def dot_strategy(ax, x, y, k, col, mk, ms):
    if k == "fixed99":   # hollow and on top: it nearly coincides with MFF for RMPL and the counting matrix
        ax.plot(x, y, marker=mk, ms=ms, mfc="none", mec=col, mew=1.1, ls="none", zorder=6)
    else:
        ax.plot(x, y, marker=mk, ms=ms, color=col, mec="white", mew=0.6, ls="none", zorder=3 if k.startswith("fixed") else 4)


def dot_panel(ax, val, ST, span, title):
    """One row per estimator, each strategy's level on a common-span axis; val(e, k) gives the level."""
    allv = [val(e, k) for e, _ in EST for k, *_ in ST]
    mid = 0.5 * (max(allv) + min(allv))
    for ei, (e, _) in enumerate(EST):
        xs = [val(e, k) for k, *_ in ST]
        ax.plot([min(xs), max(xs)], [ei, ei], color="#cfcec9", lw=1.6, zorder=2, solid_capstyle="round")
        for k, _, _, col, mk, ms in ST:
            dot_strategy(ax, val(e, k), ei, k, col, mk, ms)
    ax.set_xlim(mid - span / 2, mid + span / 2)
    ax.set_ylim(len(EST) - 0.45, -0.55)
    ax.set_title(title)
    ax.set_yticks(range(len(EST)))
    ax.set_yticklabels([e for e, _ in EST])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for ei in range(1, len(EST)):
        ax.axhline(ei - 0.5, color="#ecebe7", lw=0.6, zorder=1)


def dot_legend(fig, ST):
    h = [Line2D([], [], marker=mk, ls="none", ms=ms, mfc="none", mec=c, mew=1.1) if k == "fixed99" else
         Line2D([], [], marker=mk, ls="none", ms=ms, color=c, mec="white") for k, _, _, c, mk, ms in ST]
    fig.legend(h, [l for _, l, *_ in ST], loc="upper center", ncol=len(ST), frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0))


def box_style(face, alpha, median_lw=2.0, flier_ms=3.2):
    return dict(medianprops=dict(color="#111111", lw=median_lw), boxprops=dict(facecolor=face, edgecolor="#555555", lw=0.9, alpha=alpha),
                whiskerprops=dict(color="#555555", lw=0.9), capprops=dict(color="#555555", lw=0.9),
                flierprops=dict(marker="o", ms=flier_ms, mfc="none", mec="#555555", mew=0.8))


def np_mean_pairs(M, X):
    """Mean over pairs of each pair's mean of X over its days in mask M (pairs x days)."""
    return np.nanmean([np.nanmean(np.where(M[i], X[i], np.nan)) for i in range(len(X))])
