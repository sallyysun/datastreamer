"""Shared Chapter 6 helpers: printed width W, paths (results/ and images/ of the experiment folder the script is run
from), family labels and the paired mean +- SE used throughout. The palette and fonts live in ch6_style.py and
plot_style.py."""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_style as st  # noqa: E402  (palette, font, sizes; sets rcParams)
plt = st.plt

R, OUT = "results/", "images/"   # relative to the experiment folder each script is run from
W = st.W
INK, MUTED, GRID, SHADE, SPINE_ = st.INK, st.MUTED, st.GRID, st.SHADE, st.SPINE
EST, METHOD, BASE, THIRD = st.EST, st.METHOD, st.BASE, st.THIRD
FAMS = ["gaussian", "clayton", "gumbel", "t"]
FL = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "$t_4$"}
REG = {"constant": "Constant", "drift": "Gradual drift", "jump": "Abrupt jump"}
sfx = lambda f: "" if f == "gaussian" else "_" + f


def zero(ax, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(0, color=INK, lw=1.3, zorder=1)


def tidy(ax, grid_axis="x"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def paired(a, b):
    d = (a - b).dropna()
    return d.mean(), d.std(ddof=1) / math.sqrt(len(d))
