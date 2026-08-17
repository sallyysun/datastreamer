#!/usr/bin/env python3
"""Plot estimator tracking vs the true Spearman rho for one scenario.

Usage:
    pip install matplotlib pandas   # one-time
    python src/experiments/plot.py [scenario]   # default: jump

Reads experiment_<scenario>.csv (produced by `make experiment && ./experiment <scenario>`).
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

scenario = sys.argv[1] if len(sys.argv) > 1 else "jump"
df = pd.read_csv(f"experiment_{scenario}.csv")

plt.figure(figsize=(11, 5))
plt.plot(df["t"], df["true_rho_s"], "k-", lw=2.0, label="true rho_S")
for col in df.columns:
    if col in ("t", "true_rho_s"):
        continue
    plt.plot(df["t"], df[col], lw=0.9, alpha=0.8, label=col)

plt.xlabel("t")
plt.ylabel("Spearman rho")
plt.title(f"Online dependence tracking - {scenario}")
plt.legend(loc="best", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
out = f"experiment_{scenario}.png"
plt.savefig(out, dpi=130)
print(f"wrote {out}")
