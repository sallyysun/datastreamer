"""Download daily adjusted closes from the Yahoo Finance chart API for the 52 tickers of the 32 pairs plus ^VIX, from
2006-01-01 (period1 = 1136073600) to today. One CSV per symbol: unix,adj (adjusted close; plain close if no adjusted
series). Raw prices are not redistributed; rerun this to rebuild them. Yahoo revises adjusted history after dividends and
splits, so a fresh download is not byte-identical to the one used in the thesis (taken 2026-11-15, last day 2026-11-11).
Usage: python3 download_prices.py <prices dir>"""
import csv, json, os, sys, time, urllib.request
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T = sorted(set("JPM BAC C WFC GS MS XOM CVX COP SLB HAL OXY KO PEP PG CL MO PM HD LOW WMT TGT COST TXN ADI QCOM INTC V MA UPS "
               "FDX UNP CSX NSC PFE MRK JNJ ABT T VZ MCD YUM EWA EWC EWG EWU XLE XOP GLD GDX SPY IVV ^VIX".split()))
for s in T:
    path = f"{OUT}/{s}.csv"
    if os.path.exists(path): continue
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{s.replace('^', '%5E')}?period1=1136073600&period2=9999999999"
           "&interval=1d&events=div%2Csplits&includeAdjustedClose=true")
    r = None
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]; break
        except Exception: time.sleep(2 ** k)
    if r is None: print("FAIL", s); continue
    ts = r["timestamp"]; adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unix", "adj"]); [w.writerow([t, a]) for t, a in zip(ts, adj) if a]
    print(s, len(ts), time.strftime("%Y-%m-%d", time.gmtime(ts[0])), flush=True); time.sleep(0.3)
