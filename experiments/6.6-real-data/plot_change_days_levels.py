"""Compare predictive log loss on the top 10% of 21-day dependence changes."""
import os
import sys

import numpy as np
import pandas as pd

from real_common import EST, RC, W, change, dot_legend, dot_panel, eval_days, image, np_mean_pairs, outputs, plt, read, result, verbose_print

FIXED = [int(k) for k in os.environ.get("FIXED", "99,199,399").split(",")]
FIXSTY = {39: ("#A6A6A6", "D", 5.0), 99: ("#8a8a8a", "D", 7.5), 199: ("#6E6E6E", "s", 5.5), 399: ("#333333", "p", 6.5)}
ST = [(f"fixed{k}", rf"Fixed, $N_{{\mathrm{{eff}}}}={k}$", f"lt_fixed_{k}", *FIXSTY[k]) for k in FIXED] + [
    ("vff", r"VFF, $\alpha=0.02$", "lt_vff", "#8E6BB8", "^", 8.0), ("mff", "MFF", "lt_mffmix_t", "#C2185B", "o", 7.0)]
TAG = "" if FIXED == [99, 199, 399] else "_" + "_".join(map(str, FIXED))
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
    span = max(max(val(e, k, s) for e, _ in EST for k, *_ in ST) - min(val(e, k, s) for e, _ in EST for k, *_ in ST) for s, *_ in sets) * 1.15
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 1, figsize=(W, 3.4))
        for ax, (sk, sl, _) in zip(axs, sets):
            dot_panel(ax, lambda e, k: val(e, k, sk), ST, span, sl)
        fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.015)
        dot_legend(fig, ST)
        fig.subplots_adjust(left=0.19, right=0.975, top=0.86, bottom=0.13, hspace=0.6)
        fig.savefig(image(f"ch6_practical_final_change_days{TAG}.png"))
        plt.close(fig)
    print("CHECK MFF - fixed 199 level differences vs mff_vs_fixed199.txt (change, other): max |diff| = %.3f" % max(
        abs(val(e, "mff", s) - val(e, "fixed199", s) - r) for e, ref in REF.items() for s, r in zip(("change", "other"), ref)))
    verbose_print(Lv.pivot_table(index=["estimator", "strategy"], columns="days", values="loss").round(2).to_string())
    verbose_print(f"common span {span:.1f}")


if __name__ == "__main__":
    main()
