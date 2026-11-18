"""Compare predictive log loss on the top 10% of 21-day dependence changes."""
import os
import sys

import numpy as np
import pandas as pd

from real_common import (
    DOT_FIXED, EST, change, dot_strategies, eval_days, np_mean_pairs,
    outputs, plot_loss_levels, read, result, verbose_print,
)

FIXED = tuple(map(int, os.environ.get("FIXED", ",".join(map(str, DOT_FIXED))).split(",")))
ST = dot_strategies(FIXED)
TAG = "" if FIXED == DOT_FIXED else "_" + "_".join(map(str, FIXED))
REF = {"RMPL": (-6.93, -1.61), "Counting matrix": (-6.36, -1.13), "Hermite series": (-3.92, -3.68)}


def day_sets(S):
    roll = {p: eval_days(read(f).pipe(lambda x: x.assign(change=change(x)))).set_index("date") for p, f in outputs(S, "practical_roll2").items()}
    pairs = list(roll)
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in roll.values()])))
    CH = np.stack([roll[p].change.reindex(idx).values for p in pairs])
    present = ~np.isnan(CH)
    chg = (CH >= np.nanquantile(CH, 0.90, axis=1, keepdims=True)) & present
    return pairs, idx, [("other", "Other days", (~chg) & present), ("change", "Large estimated dependence-change days (top 10% per pair)", chg)]


def levels(S, pairs, idx, sets):
    rows = []
    for e, d in EST:
        P = {p: eval_days(read(f)).set_index("date") for p, f in outputs(S, d).items()}
        for k, _, c, *_ in ST:
            X = np.stack([-P[p][c].reindex(idx).values for p in pairs])
            rows += [dict(estimator=e, strategy=k, days=sk, loss=1e3 * np_mean_pairs(M, X)) for sk, _, M in sets]
    return pd.DataFrame(rows)


def main():
    S = sys.argv[1]
    pairs, idx, sets = day_sets(S)
    Lv = levels(S, pairs, idx, sets)
    Lv.to_csv(result(f"change_days_levels{TAG}.csv"), index=False)

    def val(e, k, s):
        return Lv[(Lv.estimator == e) & (Lv.strategy == k) & (Lv.days == s)].loss.iloc[0]
    plot_loss_levels([(key, title) for key, title, _ in sets], ST, val,
                     f"ch6_practical_final_change_days{TAG}.png",
                     height=3.4, span_padding=1.15, label_y=0.015, top=0.86, bottom=0.13, hspace=0.6)
    if 199 in FIXED:
        max_diff = max(abs(val(e, "mff", s) - val(e, "fixed199", s) - r)
                       for e, ref in REF.items() for s, r in zip(("change", "other"), ref))
        verbose_print(f"MFF − fixed 199 vs reference, max difference {max_diff:.3f}")


if __name__ == "__main__":
    main()
