"""Shared constants, loaders and plot style for the Section 6.6 scripts."""
import glob
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from config import ESTIMATORS, EST_LABEL, MEMORIES as FIX  # noqa: E402
from plot_theme import DARK, EDGE, ESTIMATOR, GRID, RC, W  # noqa: E402
from report_output import verbose_print, write_report  # noqa: E402
EST_LABELS = {e: EST_LABEL[e] if e == "rmpl" else EST_LABEL[e].capitalize() for e in ESTIMATORS}
EST = [(EST_LABELS[e], f"practical_{e}") for e in ESTIMATORS]
ECOL = {EST_LABELS[e]: ESTIMATOR[e] for e in ESTIMATORS}
EVAL_ROW, EVAL_FROM = 560, "2008-01-01"


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


def box_style(face, alpha, median_lw=2.0, flier_ms=3.2):
    return dict(medianprops=dict(color=DARK, lw=median_lw), boxprops=dict(facecolor=face, edgecolor=EDGE, lw=0.9, alpha=alpha),
                whiskerprops=dict(color=EDGE, lw=0.9), capprops=dict(color=EDGE, lw=0.9),
                flierprops=dict(marker="o", ms=flier_ms, mfc="none", mec=EDGE, mew=0.8))

