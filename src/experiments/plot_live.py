#!/usr/bin/env python3
"""Plot a capture from live_probe.py (--csv output).

The live CSV has columns: tick, true_rho_s, <estimator columns...>.

Usage (use a Python that has pandas + matplotlib -- your system python3 does):
    python3 src/experiments/plot_live.py [live.csv]

Writes <csv>.png next to the CSV and also opens a window if a display is available.
"""
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "xiao": "#ffb800",
    "xiao_ew": "#ff5555",
    "hermite": "#22d3ee",
    "hermite_ew": "#a78bfa",
    "windowed_rank": "#888888",
}

path = sys.argv[1] if len(sys.argv) > 1 else "live.csv"
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
