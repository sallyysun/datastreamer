"""Download daily adjusted closes from the Yahoo Finance chart API for the tickers of the 32 pairs plus ^VIX, from
2006-01-01 to today, one CSV per symbol (unix, adj; the plain close if there is no adjusted series). Yahoo revises
adjusted history, so a fresh download is not byte-identical to the thesis data (taken 2026-11-15, last day 2026-11-11).
Usage: python3 download_prices.py <prices dir>"""
import csv
import json
import os
import sys
import time
import urllib.request

from pairs import SYMBOLS

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


def main():
    out = sys.argv[1]
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


if __name__ == "__main__":
    main()
