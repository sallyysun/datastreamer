"""Shared Chapter 6 figure look (the palette and type of every thesis figure): estimator colours COL, the RC rc_context
dictionary, and access to plot_ch6_core (core) and matplotlib (plt). Scripts import it as `ce`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_ch6_core as core  # noqa: E402

plt = core.plt
# The first accepted look (2026-11-12): original Chapter 6 palette and type, 6.1 x 2.35 in at the printed width.
COL = {"xiao": "#2a78d6", "hermite": "#eb6834", "mle": "#1baf7a"}
INK, GRID = "#0b0b0b", "#e4e3df"
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
      "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
      "axes.titlepad": 6, "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
      "figure.dpi": 300, "savefig.dpi": 300}

