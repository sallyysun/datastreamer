#!/usr/bin/env python3
"""Xiao reference runner using the ORIGINAL onlineNPCORR code.

Imports `spearmanr_by_matrix` verbatim from external/onlineNPCORR/utility.py
(the reference estimate() formula) and drives the same online count-grid update
the reference online_corr / online_mv_corr use. Binning uses
np.searchsorted(side='right'), which is exactly the reference get_index
semantics (count of cutpoints <= x) and the C++ `upper_bound` bin rule.

Cutpoints are read from the C++ harness stderr dump, so both sides bin on
byte-identical thresholds and only the estimator arithmetic is under test.

Usage: xiao_reference.py <data.csv> <cuts.txt> [window]
       window > 0 => hard sliding window (reference online_mv_corr);
       0 (default) => cumulative (reference online_corr).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.normpath(os.path.join(HERE, "..", "..", "external", "onlineNPCORR"))
sys.path.insert(0, REF)
from utility import spearmanr_by_matrix  # noqa: E402


def load_cuts(path):
    cx = cy = None
    calib = None
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "CUTS_X":
            cx = np.array([float(v) for v in parts[1:]])
        elif parts[0] == "CUTS_Y":
            cy = np.array([float(v) for v in parts[1:]])
        elif parts[0] == "CALIB":
            calib = int(parts[1])
    return cx, cy, calib


def main():
    data = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    cuts_x, cuts_y, calib = load_cuts(sys.argv[2])
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    m1 = len(cuts_x) + 1
    m2 = len(cuts_y) + 1
    M = np.zeros((m1, m2))
    miplus = np.zeros(m1)
    mplusj = np.zeros(m2)

    ix = np.searchsorted(cuts_x, x, side="right").astype(int)
    iy = np.searchsorted(cuts_y, y, side="right").astype(int)

    out = []
    for i in range(len(x)):
        M[ix[i], iy[i]] += 1
        miplus[ix[i]] += 1
        mplusj[iy[i]] += 1
        if window > 0 and (i + 1) > window:
            j = i - window
            M[ix[j], iy[j]] -= 1
            miplus[ix[j]] -= 1
            mplusj[iy[j]] -= 1
        n = int(round(M.sum()))
        # Emit only once the C++ side is calibrated (grid holds >= calib
        # points), matching the C++ harness which returns 0.0 before then.
        if i + 1 >= calib:
            out.append(spearmanr_by_matrix(M, miplus, mplusj, n))
        else:
            out.append(0.0)
    for v in out:
        print(repr(float(v)))


if __name__ == "__main__":
    main()
