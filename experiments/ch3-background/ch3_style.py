"""Shared style and helpers for the Chapter 3 figures."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = "images"
INK, INK_2 = "#0b0b0b", "#52514e"
MARK, SHADE, GRID = "#2a78d6", "#cde2fb", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.edgecolor": INK_2, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK, "pdf.fonttype": 42, "savefig.dpi": 300})


def scatter(ax, x, y, size=3.0):
    ax.scatter(x, y, s=size, color=MARK, alpha=0.45, linewidths=0, rasterized=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def note(ax, text, loc="upper left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=INK_2, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def tail_coefficients(family, param, nu=4.0):
    """(lambda_L, lambda_U)."""
    if family == "student_t":
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - param) / (1 + param)), df=nu + 1)
        return lam, lam
    if family == "clayton":
        return 2 ** (-1 / param), 0.0
    if family == "gumbel":
        return 0.0, 2 - 2 ** (1 / param)
    return 0.0, 0.0
