"""Budget study with paired per-seed comparisons (results/budget_sweep[_TAG].csv + budget_seeds[_TAG].csv).
For each family x scenario x budget: each estimator's configuration is chosen on the validation seeds within
the memory cap; on the test seeds, DataSketches-MPL is compared with each of Xiao and Hermite and with the
better of the two, by the paired per-seed RMSE difference (W/L/T at 2 SE). Also reports the observed ranking.
Usage (from experiments/): python3 analyze_budget_paired.py [TAG]"""
import sys, math
import pandas as pd
tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
df = pd.read_csv(f"results/budget_sweep{tag}.csv")
sd = pd.read_csv(f"results/budget_seeds{tag}.csv")
BUD = {"16KiB": 16384, "64KiB": 65536, "256KiB": 262144}
E = ["xiao", "hermite", "mpl_kll", "mpl_dskll"]
val = df[df.split == "val"]
pick = {}
for (f, s), _ in df.groupby(["family", "scenario"]):
    for b, cap in BUD.items():
        for e in E:
            v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
            k = v.loc[v.rmse.idxmin()]
            pick[(f, s, b, e)] = (int(k.approx), k.gain)
def seeds(f, s, e, a, g):
    x = sd[(sd.family == f) & (sd.scenario == s) & (sd.estimator == e) & (sd.approx == a) & ((sd.gain - g).abs() < 1e-12)]
    return x.set_index("seed").rmse
for b in BUD:
    print(f"\n== {b}: test RMSE x1e-2 (mean over 100 seeds) and paired DS-MPL comparisons ==")
    rows, wlt = [], {"xiao": [0, 0, 0], "hermite": [0, 0, 0], "best_other": [0, 0, 0]}; lowest = 0
    for (f, s), _ in df.groupby(["family", "scenario"]):
        r = {e: seeds(f, s, e, *pick[(f, s, b, e)]) for e in E}
        means = {e: 100 * r[e].mean() for e in E}
        lowest += min(["xiao", "hermite", "mpl_dskll"], key=lambda e: means[e]) == "mpl_dskll"
        other = min(["xiao", "hermite"], key=lambda e: means[e])
        row = dict(family=f, scenario=s, **{e: round(means[e], 2) for e in E})
        for key, comp in [("xiao", "xiao"), ("hermite", "hermite"), ("best_other", other)]:
            dlt = 100 * (r["mpl_dskll"] - r[comp]).dropna(); m, se = dlt.mean(), dlt.std(ddof=1) / math.sqrt(len(dlt))
            wlt[key][0 if m < -2 * se else 1 if m > 2 * se else 2] += 1
            if key == "best_other": row["DS-best_other"] = f"{m:+.2f}±{2*se:.2f} ({comp})"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"DS-MPL lowest observed RMSE in {lowest}/12; paired W/L/T vs Xiao {wlt['xiao']}, vs Hermite {wlt['hermite']}, "
          f"vs the better of the two {wlt['best_other']}  (W = DS-MPL lower by > 2 paired SE)")
