#!/usr/bin/env python3
"""Figures for the memory sweep. See notes/studies/memory_sweep.md.

    ../.venv/bin/python plot_memory.py

Reads results/memory_sweep*.csv (from ./probe_memory) and writes both figures to
results/ and to notes/studies/figures/, where the note can link them (results/ is
gitignored).
"""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, os.pardir, "notes", "studies", "figures")
os.makedirs(FIGURES, exist_ok=True)


def publish(path):
    shutil.copyfile(path, os.path.join(FIGURES, os.path.basename(path)))
    print("wrote", path)

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

PANELS = [
    ("constant", "all", "constant  (target still)"),
    ("drift", "all", "drift  (slow ramp)"),
    ("oscillate", "all", "oscillate  (period 400)"),
    ("jump", "all", "jump - all ticks"),
    ("jump", "settled", "jump - settled (>1000 past)"),
    ("jump", "recovery", "jump - recovery (1000 after)"),
]

d = pd.read_csv(os.path.join(RESULTS, "memory_sweep.csv"))
ess = pd.read_csv(os.path.join(RESULTS, "memory_sweep_ess.csv"))

# The Kalman's measured half-life leaves the algebra at short memory, where a
# 20-tick block floors the response; plot it where it actually sits.
k_meas = dict(zip(ess[ess.estimator == "kalman"].param, ess[ess.estimator == "kalman"].ess_meas))


def grid(axes_title_pairs):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True)
    return fig, axes.ravel()


def style(ax, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(alpha=0.18, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)


# ---------------------------------------------------------------- figure 1
fig, axes = grid(PANELS)
for ax, (sc, win, title) in zip(axes, PANELS):
    s = d[(d.scenario == sc) & (d.window == win) & (d.estimator == "xiao_ew")].sort_values("mem_ticks")
    ax.plot(s.mem_ticks, s.mse, "-", color=INK, lw=2.0, marker="o", ms=4.0, label="MSE")
    ax.plot(s.mem_ticks, s.bias2, "--", color=ORANGE, lw=2.0, marker="s", ms=4.0, label="bias$^2$")
    ax.plot(s.mem_ticks, s["var"], ":", color=BLUE, lw=2.0, marker="^", ms=4.5, label="variance")
    best = s.loc[s.mse.idxmin()]
    ax.axvline(best.mem_ticks, color=MUTED, lw=1.0, ls="-", alpha=0.55)
    style(ax, title)
    ax.set_ylim(top=ax.get_ylim()[1] * 4.0)
    ax.text(0.035, 0.96, f"optimum {best.mem_ticks:,.0f} ticks\nRMSE {best.rmse:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
for ax in axes[3:]:
    ax.set_xlabel("effective memory (ticks)", fontsize=9, color=MUTED)
for ax in (axes[0], axes[3]):
    ax.set_ylabel(r"error$^2$  (Spearman $\rho$ units$^2$)", fontsize=9, color=MUTED)
axes[0].legend(fontsize=9, frameon=False, loc="lower left")
fig.suptitle("Figure 1  -  where the optimum sits depends on how fast the target moves "
             "(xiao_ew, 40 seeds, N=5000)", fontsize=12, color=INK, x=0.012, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.955))
out1 = os.path.join(RESULTS, "memory_fig1_ucurve.png")
fig.savefig(out1, dpi=140)
publish(out1)

# ---------------------------------------------------------------- figure 2
SERIES = [("xiao_ew", BLUE, "-", "o"), ("hermite_ew", ORANGE, "--", "s"), ("kalman", AQUA, ":", "^")]
fig, axes = grid(PANELS)
for ax, (sc, win, title) in zip(axes, PANELS):
    for est, col, ls, mk in SERIES:
        s = d[(d.scenario == sc) & (d.window == win) & (d.estimator == est)].sort_values("mem_ticks")
        x = np.array([k_meas.get(p, m) for p, m in zip(s.param, s.mem_ticks)]) if est == "kalman" \
            else s.mem_ticks.values
        ax.errorbar(x, s.rmse, yerr=s.se_rmse, color=col, ls=ls, lw=2.0, marker=mk, ms=4.2,
                    capsize=2.0, elinewidth=1.0, label=est)
    ax.axvspan(500, 40000, color=MUTED, alpha=0.06, lw=0)
    style(ax, title)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
axes[0].text(0.40, 0.96, "shaded: hermite_ew not warm within N=5000",
             transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
axes[4].text(0.035, 0.96, "kalman's 20-tick block floors its\nmemory near 40 ticks (leftmost 2 points)",
             transform=axes[4].transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
for ax in axes[3:]:
    ax.set_xlabel("effective memory (ticks)", fontsize=9, color=MUTED)
for ax in (axes[0], axes[3]):
    ax.set_ylabel(r"RMSE (Spearman $\rho$)", fontsize=9, color=MUTED)
axes[0].legend(fontsize=9, frameon=False, loc="lower left")
fig.suptitle("Figure 2  -  three different algorithms, one memory axis "
             "(40 seeds, error bars = SE across seeds)", fontsize=12, color=INK, x=0.012, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.955))
out2 = os.path.join(RESULTS, "memory_fig2_families.png")
fig.savefig(out2, dpi=140)
publish(out2)
