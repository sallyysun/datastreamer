"""Build the Section 6.6 runner inputs from the downloaded adjusted closes: for each pair, daily log returns on common
trading days, each divided by its EWMA(0.94) volatility forecast from the previous day (variance seeded with the first
20 returns). Writes <out>/in/<A>_<B>.csv (date, x, y) and <out>/vix.csv.
Usage: python3 build_inputs.py <prices dir> <out dir>"""
import os
import sys

import numpy as np
import pandas as pd

from pairs import PAIRS


def load(prices, s):
    return pd.read_csv(f"{prices}/{s}.csv").set_index("unix").adj.rename(s)


def ewvol(r):
    v = np.empty_like(r)
    v[0] = np.var(r[:20])
    for t in range(1, len(r)):
        v[t] = 0.94 * v[t - 1] + 0.06 * r[t - 1] ** 2
    return np.sqrt(v)


def main():
    prices, out = sys.argv[1], sys.argv[2]
    os.makedirs(f"{out}/in", exist_ok=True)
    for A, B in PAIRS:
        px = pd.concat([load(prices, A), load(prices, B)], axis=1).dropna()
        dates = pd.to_datetime(px.index[1:], unit="s").strftime("%Y-%m-%d")
        r = np.diff(np.log(px.values), axis=0)
        z = np.column_stack([r[:, 0] / ewvol(r[:, 0]), r[:, 1] / ewvol(r[:, 1])])
        pd.DataFrame({"date": dates, "x": z[:, 0], "y": z[:, 1]}).to_csv(f"{out}/in/{A}_{B}.csv", index=False, float_format="%.10g")
    v = load(prices, "^VIX")
    v.index = pd.to_datetime(v.index, unit="s").strftime("%Y-%m-%d")
    v.rename("vix").to_csv(f"{out}/vix.csv")


if __name__ == "__main__":
    main()
