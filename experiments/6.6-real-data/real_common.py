"""Shared constants, loaders and plot style for the Section 6.6 real-data scripts."""
import glob
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from config import EST_COLOUR_KEY, ESTIMATORS, EST_LABEL, MEMORIES as FIX, STRATEGY_COLOUR  # noqa: E402
from plot_theme import COL, GRID, INK, RC, W  # noqa: E402
from report_output import verbose_print, write_report  # noqa: E402
EST_LABELS = {e: EST_LABEL[e] if e == "rmpl" else EST_LABEL[e].capitalize() for e in ESTIMATORS}
EST = [(EST_LABELS[e], f"practical_{e}") for e in ESTIMATORS]
ECOL = {EST_LABELS[e]: COL[EST_COLOUR_KEY[e]] for e in ESTIMATORS}
EVAL_ROW, EVAL_FROM = 560, "2008-01-01"
DOT_FIXED = (99, 199, 399)
DOT_FIXED_STYLE = {39: ("#A6A6A6", "D", 5.0), 99: ("#8a8a8a", "D", 7.5),
                   199: ("#6E6E6E", "s", 5.5), 399: ("#333333", "p", 6.5)}


def result(name):
    return os.path.join(HERE, "results", name)


def image(name):
    return os.path.join(HERE, "images", name)


def outputs(S, d):
    """{pair: path} of the runner outputs in S/d/out, sorted by pair."""
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


def dot_strategies(fixed=DOT_FIXED):
    return [(f"fixed{k}", rf"Fixed, $N_{{\mathrm{{eff}}}}={k}$", f"lt_fixed_{k}", *DOT_FIXED_STYLE[k])
            for k in fixed] + [
        ("vff", r"VFF, $\alpha=0.02$", "lt_vff", STRATEGY_COLOUR["vff_slow"], "^", 8.0),
        ("mff", "MFF", "lt_mffmix_t", STRATEGY_COLOUR["mff"], "o", 7.0),
    ]


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


def plot_loss_levels(conditions, strategies, loss, out, *, height, span_padding, label_y, top, bottom, hspace):
    """Plot loss by estimator and strategy for each condition."""
    ranges = []
    for condition, _ in conditions:
        values = [loss(e, k, condition) for e, _ in EST for k, *_ in strategies]
        ranges.append(max(values) - min(values))
    span = max(ranges) * span_padding

    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(conditions), 1, figsize=(W, height), squeeze=False)
        for ax, (condition, title) in zip(axes[:, 0], conditions):
            dot_panel(ax, lambda e, k: loss(e, k, condition), strategies, span, title)
        fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=label_y)
        dot_legend(fig, strategies)
        fig.subplots_adjust(left=0.19, right=0.975, top=top, bottom=bottom, hspace=hspace)
        fig.savefig(image(out))
        plt.close(fig)


def box_style(face, alpha, median_lw=2.0, flier_ms=3.2):
    return dict(medianprops=dict(color="#111111", lw=median_lw), boxprops=dict(facecolor=face, edgecolor="#555555", lw=0.9, alpha=alpha),
                whiskerprops=dict(color="#555555", lw=0.9), capprops=dict(color="#555555", lw=0.9),
                flierprops=dict(marker="o", ms=flier_ms, mfc="none", mec="#555555", mew=0.8))


def np_mean_pairs(M, X):
    """Mean over pairs of each pair's mean of X over its days in mask M (pairs x days)."""
    return np.nanmean([np.nanmean(np.where(M[i], X[i], np.nan)) for i in range(len(X))])
