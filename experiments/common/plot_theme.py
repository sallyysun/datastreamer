"""Chapter 6 figure styles, colours and helpers, without Matplotlib state changes."""
import os

import matplotlib

W = 6.1
INK, GRID = "#0b0b0b", "#e4e3df"
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
      "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
      "axes.titlepad": 6, "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
      "figure.dpi": 300, "savefig.dpi": 300}

DARK = "#222222"    # zero lines, medians, change points
TEXT = "#333333"    # annotations
EDGE = "#444444"    # box edges, whiskers, error bars
MUTED = "#52514e"   # secondary text and markers
REF = "#9a9a9a"     # reference and guide lines
LIGHT = "#cfcfcf"   # de-emphasised bars and ranges

ESTIMATOR = {"rmpl": "#1baf7a", "xiao": "#2a78d6", "hermite": "#eb6834"}
STRATEGY = {"fixed": TEXT, "fixed_other": LIGHT, "vff": "#8E6BB8", "mff": "#C2185B"}
FIXED_GREY = {39: "#A6A6A6", 99: "#8a8a8a", 199: "#6E6E6E", 399: TEXT}
FAMILY = {"gaussian": REF, "t": "#C2185B", "clayton": "#8E6BB8", "gumbel": "#E3B505"}
ORANGE, BLUE = "#E69F00", "#0072B2"
HEDGE, FIXED_SHARE = ORANGE, BLUE
KLL_OWN = "#6250d6"
REGRET = {"static": "#3b4f7a", "switching": "#9fb0cf"}
SQUARED_BETTER, LOG_BETTER = "#a51c30", "#1d4f91"
MEMORY_POS = (0.82, 0.64, 0.47, 0.30, 0.10)


def memory_colours(memories):
    cmap = matplotlib.colormaps["plasma"]
    return dict(zip(memories, (cmap(p) for p in MEMORY_POS)))


def grid(ax, axis="both", which="major"):
    ax.grid(True, axis=axis, which=which, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, out="images/"):
    import matplotlib.pyplot as plt
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def swatch(color=None, **kw):
    from matplotlib.patches import Rectangle
    return Rectangle((0, 0), 1, 1, color=color, **kw)


def top_legend(fig, handles, labels, x=0.53, y=1.0, fontsize=7, **kw):
    return fig.legend(handles, labels, loc="upper center", frameon=False, fontsize=fontsize, bbox_to_anchor=(x, y), **kw)


def bottom_label(fig, text, x=0.53, y=0.01, fontsize=8):
    fig.text(x, y, text, ha="center", fontsize=fontsize)
