"""Tables for probe_batch_ise: rho_S RMSE (x1e-3) and copula CDF ISE (x1e-4) of the family bank against the Chapter 6
banks, its own best fixed arm, and the offline batch estimators (full sample, oracle segment, two-sided kernel with
the best bandwidth in hindsight). Mean over seeds; W/L/T = family bank lower / higher / within 2 SE (paired).
Note: notes/studies/thesis_bank_ise_batch.md.   Usage (from experiments/): python3 batch_ise_table.py [first_seed=42]"""
import sys
import pandas as pd

first = sys.argv[1] if len(sys.argv) > 1 else "42"
fams = ["gaussian", "clayton", "gumbel", "t", "gauss2t", "clayton2gumbel", "gauss2clayton"]
d = pd.concat(pd.read_csv(f"results/batch_ise_{first}{'' if c == 'gaussian' else '_' + c}.csv") for c in fams)
online = ["xiao_bank_log", "mle_bank_log", "xiao_bank_own", "hermite_bank_own"]


def cell_table(metric, win, scale):
    w = d[(d.window == win) & (d.metric == metric)].pivot_table(index=["copula", "scenario", "seed"], columns="method",
                                                               values="value") * scale
    rows, cnt = [], {}
    for (c, s), g in w.groupby(level=[0, 1], sort=False):
        means = g.mean()
        best_arm = means[[m for m in means.index if m.startswith("arm_")]].idxmin()
        best_k = means[[m for m in means.index if m.startswith("batch_kernel_")]].idxmin()
        refs = online + [best_arm, "batch_full", "batch_segment", best_k]
        cells = [f"{means.family_bank:.2f}"]
        for k, r in zip(online + ["best arm", "batch full", "batch segment", "batch kernel"], refs):
            x = (g.family_bank - g[r]).dropna()
            h = 2 * x.std(ddof=1) / len(x) ** .5
            res = "W" if x.mean() < -h else "L" if x.mean() > h else "T"
            cnt.setdefault(k, {"W": 0, "L": 0, "T": 0})[res] += 1
            extra = f" ({r[4:]})" if k == "best arm" else f" (h={r.split('_')[-1]})" if k == "batch kernel" else ""
            cells.append(f"{means[r]:.2f}{extra}")
        rows.append(f"| {c} | {s} | " + " | ".join(cells) + " |")
    return rows, cnt


for metric, scale, unit in [("rmse", 1000, "rho_S RMSE x1e-3"), ("ise_x1e4", 1, "copula ISE x1e-4")]:
    for win in ["whole", "rec_0_100", "rec_100_500"]:
        rows, cnt = cell_table(metric, win, scale)
        if not rows: continue
        print(f"\n## {unit}, window {win} (mean over seeds)\n")
        print("| copula | scenario | family bank | xiao_log | mle_log | xiao_own | hermite_own | best arm | batch full | "
              "batch segment (oracle) | batch kernel (best h) |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        print("\n".join(rows))
        print("\nFamily bank W/L/T (paired, 2 SE): " + "; ".join(f"{k} {v['W']}/{v['L']}/{v['T']}" for k, v in cnt.items()))
