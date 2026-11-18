"""Section 6.6 inputs for the 32 pre-specified pairs.
  python3 inputs.py download <prices dir>         daily adjusted closes from the Yahoo Finance chart API, 2006-01-01 to
                                                  today, one CSV per symbol (unix, adj; the plain close if no adjusted series)
  python3 inputs.py build <prices dir> <out dir>  per pair, log returns on common trading days divided by the previous day's
                                                  EWMA(0.94) volatility (variance seeded with the first 20 returns):
                                                  <out>/in/<A>_<B>.csv (date, x, y) and <out>/vix.csv
Yahoo revises adjusted history, so a fresh download is not byte-identical to the thesis data (taken 2026-11-15, last day
2026-11-11)."""
import csv
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

PAIRS = [("JPM", "BAC"), ("JPM", "C"), ("BAC", "C"), ("WFC", "JPM"), ("GS", "MS"), ("XOM", "CVX"), ("COP", "XOM"), ("SLB", "HAL"),
         ("OXY", "COP"), ("KO", "PEP"), ("PG", "CL"), ("MO", "PM"), ("HD", "LOW"), ("WMT", "TGT"), ("COST", "WMT"), ("TXN", "ADI"),
         ("QCOM", "TXN"), ("INTC", "TXN"), ("V", "MA"), ("UPS", "FDX"), ("UNP", "CSX"), ("NSC", "CSX"), ("PFE", "MRK"), ("JNJ", "PFE"),
         ("ABT", "JNJ"), ("T", "VZ"), ("MCD", "YUM"), ("EWA", "EWC"), ("EWG", "EWU"), ("XLE", "XOP"), ("GLD", "GDX"), ("SPY", "IVV")]
SYMBOLS = sorted({s for p in PAIRS for s in p} | {"^VIX"})
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=1136073600&period2=9999999999"
       "&interval=1d&events=div%2Csplits&includeAdjustedClose=true")


def fetch(s):
    req = urllib.request.Request(URL.format(s.replace("^", "%5E")), headers={"User-Agent": "Mozilla/5.0"})
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
        except Exception:
            time.sleep(2 ** k)
    return None


def download(out):
    os.makedirs(out, exist_ok=True)
    for s in SYMBOLS:
        path = f"{out}/{s}.csv"
        if os.path.exists(path):
            continue
        r = fetch(s)
        if r is None:
            print("FAIL", s, file=sys.stderr)
            continue
        ts = r["timestamp"]
        adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unix", "adj"])
            w.writerows([t, a] for t, a in zip(ts, adj) if a)
        time.sleep(0.3)


def load(prices, s):
    return pd.read_csv(f"{prices}/{s}.csv").set_index("unix").adj.rename(s)


def ewvol(r):
    v = np.empty_like(r)
    v[0] = np.var(r[:20])
    for t in range(1, len(r)):
        v[t] = 0.94 * v[t - 1] + 0.06 * r[t - 1] ** 2
    return np.sqrt(v)


def build(prices, out):
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


def main():
    command, *args = sys.argv[1:]
    {"download": download, "build": build}[command](*args)


if __name__ == "__main__":
    main()
