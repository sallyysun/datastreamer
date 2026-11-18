"""Shared styles for the storage-budget figures."""
import config as C
from plot_theme import COL


def budget_estimators(labels=C.EST_LABEL):
    return [
        ("mpl_dskll" if e == "rmpl" else e, labels[e], COL[C.EST_COLOUR_KEY[e]], C.EST_MARKER[e])
        for e in C.ESTIMATORS
    ]


def log2_axis(set_scale, set_ticks, set_labels, ticks):
    set_scale("log", base=2)
    set_ticks(ticks)
    set_labels([f"{v:g}" for v in ticks])


def budget_guides(ax):
    for budget in C.BUDGETS:
        ax.axvline(budget, color="#8a8984", lw=0.5, ls=":", zorder=0)
