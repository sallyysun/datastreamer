"""Build the probe inputs from the downloaded adjusted closes (practical application, Chapter 6.6).
For each of the 32 pre-specified pairs: common trading days, daily log returns, each divided by its EWMA(0.94) volatility
forecast from the previous day (variance seeded with the first 20 returns). Writes <out>/in/<A>_<B>.csv (date, x, y) and
Usage: python3 build_inputs.py <prices dir from 01_download_prices.py> <out dir>"""
import os, sys
import numpy as np, pandas as pd
PRICES, OUT = sys.argv[1], sys.argv[2]
PAIRS = [("JPM","BAC"),("JPM","C"),("BAC","C"),("WFC","JPM"),("GS","MS"),("XOM","CVX"),("COP","XOM"),("SLB","HAL"),("OXY","COP"),
         ("KO","PEP"),("PG","CL"),("MO","PM"),("HD","LOW"),("WMT","TGT"),("COST","WMT"),("TXN","ADI"),("QCOM","TXN"),("INTC","TXN"),
         ("V","MA"),("UPS","FDX"),("UNP","CSX"),("NSC","CSX"),("PFE","MRK"),("JNJ","PFE"),("ABT","JNJ"),("T","VZ"),("MCD","YUM"),
         ("EWA","EWC"),("EWG","EWU"),("XLE","XOP"),("GLD","GDX"),("SPY","IVV")]
def load(s): return pd.read_csv(f"{PRICES}/{s}.csv").set_index("unix").adj.rename(s)
def ewvol(r):
    v = np.empty_like(r); v[0] = np.var(r[:20])
    for t in range(1, len(r)): v[t] = 0.94 * v[t - 1] + 0.06 * r[t - 1] ** 2
    return np.sqrt(v)
os.makedirs(f"{OUT}/in", exist_ok=True)
for A, B in PAIRS:
    px = pd.concat([load(A), load(B)], axis=1).dropna(); dates = pd.to_datetime(px.index[1:], unit="s").strftime("%Y-%m-%d")
    r = np.diff(np.log(px.values), axis=0)
    z = np.column_stack([r[:, 0] / ewvol(r[:, 0]), r[:, 1] / ewvol(r[:, 1])])
    pd.DataFrame({"date": dates, "x": z[:, 0], "y": z[:, 1]}).to_csv(f"{OUT}/in/{A}_{B}.csv", index=False, float_format="%.10g")
v = load("^VIX"); v.index = pd.to_datetime(v.index, unit="s").strftime("%Y-%m-%d"); v.rename("vix").to_csv(f"{OUT}/vix.csv")
print("ok", len(PAIRS))
