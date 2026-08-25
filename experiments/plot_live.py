#!/usr/bin/env python3
"""Plot a capture from live_probe.py (--csv output).

The live CSV has columns: tick, true_rho_s, <estimator columns...>.

Usage (use a Python that has pandas + matplotlib -- your system python3 does):
    python3 experiments/plot_live.py [live.csv]

A bare filename is resolved against experiments/results/. Writes <csv>.png next to the CSV.
"""
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLORS = {
    "xiao": "#ffb800",
    "xiao_ew": "#ff5555",
    "hermite": "#22d3ee",
    "hermite_ew": "#a78bfa",
    "windowed_rank": "#888888",
    "gaussian_mle": "#2e7d32",
    "gaussian_mle_ew": "#00e676",
}

path = sys.argv[1] if len(sys.argv) > 1 else "live.csv"
if not os.path.isabs(path) and not os.path.exists(path):
    path = os.path.join(RESULTS, path)
df = pd.read_csv(path)

x = "tick" if "tick" in df.columns else "t"
skip = {x, "true_rho_s"}

plt.figure(figsize=(11, 4.5))
plt.plot(df[x], df["true_rho_s"], color="black", lw=2.4, label="true rho_S")
for col in df.columns:
    if col in skip:
        continue
    plt.plot(df[x], df[col], lw=1.2, alpha=0.85, color=COLORS.get(col), label=col)

plt.xlabel(x)
plt.ylabel("Spearman rho")
plt.ylim(-1, 1)
plt.title(f"Live online dependence tracking — {os.path.basename(path)}")
plt.legend(loc="best", fontsize=8, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()

out = os.path.splitext(path)[0] + ".png"
plt.savefig(out, dpi=130)
print(f"wrote {out}")
plt.show()
