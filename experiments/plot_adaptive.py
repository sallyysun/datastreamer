#!/usr/bin/env python3
"""Figures for the adaptive-memory experiment. See notes/studies/adaptive_memory.md.

    ../.venv/bin/python plot_adaptive.py

Reads results/adaptive_memory*.csv (from ./probe_adaptive) and writes both figures
to results/ and to notes/studies/figures/, where the note links them (results/ is
gitignored).
"""
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, os.pardir, "notes", "studies", "figures")
os.makedirs(FIGURES, exist_ok=True)

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c0392b"

FAMS = [("xiao", BLUE, "Xiao bank"),
        ("hermite_db", ORANGE, "Hermite bank (debiased)"),
        ("kalman", AQUA, "Kalman bank")]


def publish(path):
    shutil.copyfile(path, os.path.join(FIGURES, os.path.basename(path)))
    print("wrote", path)


def style(ax, title, logy=False):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(alpha=0.18, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    if logy:
        ax.set_yscale("log")


traj = pd.read_csv(os.path.join(RESULTS, "adaptive_memory_traj.csv"))
pairs = pd.read_csv(os.path.join(RESULTS, "adaptive_memory_pairs.csv"))

# ------------------------------------------------- figure 1: the climax figure
j = traj[traj.scenario == "jump"]
fig, (top, bot) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.35, 1.0]})

for fam, col, label in FAMS:
    s = j[(j.family == fam) & (j.method == "bank")].sort_values("t")
    top.plot(s.t, s.mem_post, "-", color=col, lw=2.0, label=label)
s = j[(j.family == "xiao") & (j.method == "vff_slow")].sort_values("t")
top.plot(s.t, s.mem_post, "--", color=RED, lw=1.6, label="Xiao VFF (rate 0.05)")

top.axvline(2500, color=MUTED, lw=1.0, ls=":")
top.set_ylim(28, 4200)
top.text(2530, 2400, "jump", fontsize=9, color=MUTED)
top.set_ylabel("posterior-mean memory (ticks)", fontsize=9.5, color=INK)
style(top, "Memory chosen by the estimator, through a jump at t = 2500  "
           "(mean over 40 seeds)", logy=True)
top.legend(fontsize=9, frameon=False, loc="upper left")

for fam, col, label in FAMS:
    s = j[(j.family == fam) & (j.method == "bank")].sort_values("t")
    bot.plot(s.t, s.rho_hat, "-", color=col, lw=1.7, label=label)
s = j[(j.family == "xiao") & (j.method == "bank")].sort_values("t")
bot.plot(s.t, s.truth, color=INK, lw=1.3, ls="--", label="true $\\rho_S$")
bot.axvline(2500, color=MUTED, lw=1.0, ls=":")
bot.set_xlabel("tick", fontsize=9.5, color=INK)
bot.set_ylabel("$\\hat{\\rho}_S$", fontsize=10, color=INK)
style(bot, "and what that buys in the estimate")
bot.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)

fig.tight_layout()
p = os.path.join(RESULTS, "adaptive_fig1_trajectory.png")
fig.savefig(p, dpi=170)
publish(p)

# ------------------------------------------------- figure 2: the 3x4 table
SCEN = ["constant", "drift", "jump", "oscillate"]
kb = pairs[pairs.baseline == "known_best"]
fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0), sharey=True)
width = 0.26
for ax, sc in zip(axes, SCEN):
    for i, (fam, col, label) in enumerate(FAMS):
        s = kb[(kb.family == fam) & (kb.scenario == sc)]
        base = float(s[s.method == "known_best"].rmse.iloc[0])
        bank = float(s[s.method == "bank"].rmse.iloc[0])
        vff = s[s.method.str.startswith("vff")].rmse.min()
        for k, (v, hatch) in enumerate([(base, ""), (vff, "//"), (bank, "xx")]):
            ax.bar(i + (k - 1) * width, v, width * 0.92, color=col,
                   alpha=[0.35, 0.62, 1.0][k], hatch=hatch, edgecolor="white", lw=0.6)
        sig = float(s[s.method == "bank"].sigma.iloc[0])
        ax.text(i + width, bank, "%+.1f$\\sigma$" % sig, ha="center", va="bottom",
                fontsize=8.0, color=INK if sig < 0 else RED)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f[0].replace("hermite_db", "hermite") for f in FAMS], fontsize=9)
    style(ax, sc)
axes[0].set_ylabel("RMSE  (left: known_best oracle, middle: best VFF, right: bank)",
                   fontsize=9, color=INK)
fig.tight_layout()
p = os.path.join(RESULTS, "adaptive_fig2_table.png")
fig.savefig(p, dpi=170)
publish(p)
