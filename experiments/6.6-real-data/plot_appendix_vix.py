"""Compare predictive loss by VIX regime."""
import sys

import numpy as np
import pandas as pd

from real_common import EST, dot_strategies, eval_days, outputs, plot_loss_levels, read, result, verbose_print

ST = dot_strategies()
REG = [("calm", "Calm (VIX < 20)"), ("mid", "Moderate (20 ≤ VIX < 25)"), ("stress", "Stress (VIX ≥ 25)")]
REF = {"RMPL": (-1.79, -2.65, -8.67), "Counting matrix": (-2.40, -1.56, -5.94), "Hermite series": (-3.64, -2.45, -5.75)}


def levels(S, vix):
    rows, ndays = [], {}
    for e, d in EST:
        for p, f in outputs(S, d).items():
            x = eval_days(read(f))
            v = vix.reindex(pd.DatetimeIndex(x.date)).ffill().values
            assert not np.isnan(v).any()
            for rk, m in {"calm": v < 20, "mid": (v >= 20) & (v < 25), "stress": v >= 25}.items():
                if e == "RMPL":
                    ndays[rk] = ndays.get(rk, 0) + int(m.sum())
                rows += [dict(estimator=e, pair=p, regime=rk, strategy=k, loss=-1e3 * x[c].values[m].mean()) for k, _, c, *_ in ST]
    return pd.DataFrame(rows).groupby(["estimator", "regime", "strategy"]).loss.mean().reset_index(), ndays


def main():
    vix = pd.read_csv(sys.argv[2], index_col=0).iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    Lv, ndays = levels(sys.argv[1], vix)
    Lv.to_csv(result("vix_levels.csv"), index=False)

    def val(e, k, r):
        return Lv[(Lv.estimator == e) & (Lv.regime == r) & (Lv.strategy == k)].loss.iloc[0]
    max_diff = max(abs(round(val(e, "mff", r) - val(e, "vff", r), 2) - ref[i])
                   for e, ref in REF.items() for i, (r, _) in enumerate(REG))
    verbose_print(f"MFF − VFF vs reference, max difference {max_diff:.3f}")
    plot_loss_levels(REG, ST, val, "ch6_practical_appendix_vix.png",
                     height=4.6, span_padding=1.12, label_y=0.012, top=0.9, bottom=0.1, hspace=0.62)
    verbose_print("VIX pair-days", ndays)


if __name__ == "__main__":
    main()
