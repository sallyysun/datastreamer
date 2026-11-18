"""Chapter 6 Matplotlib defaults and shared figure styles."""
from plot_theme import *  # noqa: F401,F403

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R, OUT = "results/", "images/"

# Global defaults under RC: the settings RC does not override (tick widths, mathtext tick labels, legend and grid) still
# shape the thesis figures.
_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "serif", "font.serif": [f for f in ("STIX Two Text", "STIXGeneral") if f in _have] or ["DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "font.size": 9.5, "axes.titlesize": 12, "axes.titleweight": "normal", "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5, "figure.titlesize": 12,
    "text.color": "#333333", "axes.labelcolor": "#333333", "axes.titlecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.edgecolor": "#9A9A9A", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
    "axes.grid": False, "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
})
