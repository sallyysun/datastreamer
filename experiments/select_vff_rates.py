"""VFF controller rates from development runs (probe_vff_common, seeds 1-30), by a fixed rule:
slow = rate with the lowest mean whole-run RMSE over constant, drift and oscillate;
fast = rate with the lowest mean RMSE over the 500 ticks after the jump.
Means over estimators, families and seeds. Usage: python3 select_vff_rates.py [TAG]"""
import sys
import pandas as pd
tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
d = pd.concat(pd.read_csv(f"results/vff_common{tag}{s}.csv") for s in ["", "_clayton", "_gumbel", "_t"])
smooth = d[(d.window == "whole") & d.scenario.isin(["constant", "drift", "oscillate"])].groupby("rate").rmse.mean()
jump = d[(d.window == "post500") & (d.scenario == "jump")].groupby("rate").rmse.mean()
print("smooth (constant+drift+oscillate, whole) RMSE x1000:", (1000 * smooth).round(2).to_dict())
print("jump post-500 RMSE x1000:", (1000 * jump).round(2).to_dict())
print(f"slow = {smooth.idxmin()}, fast = {jump.idxmin()}")
