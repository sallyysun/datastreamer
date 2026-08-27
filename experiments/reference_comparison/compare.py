#!/usr/bin/env python3
"""Load two per-step estimate files and report divergence stats.

Usage: compare.py <a.txt> <b.txt> [skip] [label]
`skip` drops the leading rows (warm-up / gated region) before scoring.
"""
import sys
import numpy as np

a = np.array([float(l) for l in open(sys.argv[1])])
b = np.array([float(l) for l in open(sys.argv[2])])
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
label = sys.argv[4] if len(sys.argv) > 4 else ""

assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
d = np.abs(a - b)[skip:]
i = int(d.argmax()) + skip
print(f"{label:22s} n={len(a):5d} cmp={len(d):5d}  "
      f"max={d.max():.3e}  mean={d.mean():.3e}  "
      f"(worst @{i}: {a[i]:.12g} vs {b[i]:.12g})")
