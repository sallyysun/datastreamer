"""Matched bank-vs-alternative table from probe_adaptive's per-seed RMSE.

Paired over seeds; 95% t-interval (t_39). Writes results/matched_table.csv.
Run from experiments/ after ./probe_adaptive. See notes/studies/vff_norm.md.
"""
import numpy as np
import pandas as pd

T39 = 2.0227
SCENARIOS = ["constant", "drift", "jump", "oscillate"]

seeds = pd.read_csv("results/adaptive_memory_seeds.csv")
summary = pd.read_csv("results/adaptive_memory.csv")


def series(family, method, scenario, param=None):
    x = seeds[(seeds.family == family) & (seeds.method == method) & (seeds.scenario == scenario)]
    if param is not None:
        x = x[np.isclose(x.param, param)]
    return x.sort_values("seed").rmse.values


fixed = summary[summary.method == "fixed"]
rows = []
for family in ["xiao", "hermite_db", "kalman"]:
    fx = fixed[fixed.family == family]
    one = fx.groupby("param").rmse.mean().idxmin()
    for scenario in SCENARIOS:
        best = fx[fx.scenario == scenario].sort_values("rmse").iloc[0].param
        bank = series(family, "bank", scenario)
        others = {
            "one_setting": series(family, "fixed", scenario, one),
            "best_fixed_post_hoc": series(family, "fixed", scenario, best),
            "vff_slow": series(family, "vff_slow", scenario),
            "vff_fast": series(family, "vff_fast", scenario),
        }
        for name, other in others.items():
            d = bank - other
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            rows.append((family, scenario, name, other.mean(), bank.mean(), m,
                         m - T39 * se, m + T39 * se, 100 * m / other.mean()))

table = pd.DataFrame(rows, columns=["family", "scenario", "vs", "rmse_other", "rmse_bank",
                                    "diff", "ci_lo", "ci_hi", "pct"])
table.to_csv("results/matched_table.csv", index=False)
for family in table.family.unique():
    print("==", family)
    print(table[table.family == family].pivot(index="vs", columns="scenario", values="pct").round(1))
