"""Shared Chapter 6 figure style: one palette, one font, one set of sizes for every ch6 script.
Font: CH6_FONT=stix (default; STIX Two Text if installed, else STIXGeneral, with STIX maths) or CH6_FONT=cm (Computer
Modern, to match a thesis set in default LaTeX fonts). Sizes are for figures saved at the printed width W."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

W = 6.1   # printed text width, inches

# estimators (fixed across every Chapter 6 figure)
XIAO, HERMITE, RMPL = "#647F91", "#A47D8D", "#8A9472"
EST = {"xiao": XIAO, "hermite": HERMITE, "mle": RMPL, "mpl_dskll": RMPL, "mpl_kll": RMPL}
EST_LABEL = {"xiao": "Counting matrix", "hermite": "Hermite series", "mle": "RMPL", "mpl_dskll": "RMPL"}
EST_MARKER = {"xiao": "o", "hermite": "s", "mle": "^", "mpl_dskll": "^"}   # markers carry identity where hue is weak

# neutrals
INK, MUTED, GRID, SPINE, SHADE, BG = "#333333", "#6B6B6B", "#E5E5E5", "#9A9A9A", "#F3F3F3", "#FFFFFF"

# procedures (not estimators): method of interest, its baseline, a third comparator
METHOD, BASE, THIRD, FOURTH = "#3E4A56", "#B3B3B3", "#C29B57", "#7C9A9C"

# bias / variance / MSE for a single estimator: MSE in the estimator's colour, the parts in style + neutral hues
BIAS2 = dict(color="#C29B57", ls=(0, (4, 2)), lw=1.3)
VAR = dict(color="#8C8C8C", ls=(0, (1, 1.6)), lw=1.5)

# ordered ramps (short -> long memory; small -> large budget)
MEM_RAMP = ["#D6DDE2", "#AEBBC5", "#8799A7", "#647F91", "#3F5566"]
BUDGET_RAMP = {16: "#AEBBC5", 64: "#647F91", 256: "#3F5566"}

TITLE, LABEL, TICK, ANN, SMALL = 12, 11, 10, 9.5, 8.5

FONT = os.environ.get("CH6_FONT", "stix")


def _serif():
    have = {f.name for f in font_manager.fontManager.ttflist}
    if FONT == "cm":
        return ["cmr10"], "cm"
    return [f for f in ("STIX Two Text", "STIXGeneral") if f in have] or ["DejaVu Serif"], "stix"


def apply():
    serif, math = _serif()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": serif, "mathtext.fontset": math, "axes.formatter.use_mathtext": True,
        "font.size": ANN, "axes.titlesize": TITLE, "axes.titleweight": "normal", "axes.labelsize": LABEL,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": ANN, "figure.titlesize": TITLE,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 6,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    })


apply()
