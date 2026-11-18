"""Compare predictive loss by VIX regime."""
import sys

import numpy as np
import pandas as pd

from real_common import EST, RC, W, dot_legend, dot_panel, eval_days, image, outputs, plt, read, result, verbose_print

ST = [("fixed99", r"Fixed, $N_{\mathrm{eff}}=99$", "lt_fixed_99", "#8a8a8a", "D", 7.5),
      ("fixed199", r"Fixed, $N_{\mathrm{eff}}=199$", "lt_fixed_199", "#6E6E6E", "s", 5.5),
      ("fixed399", r"Fixed, $N_{\mathrm{eff}}=399$", "lt_fixed_399", "#333333", "p", 6.5),
      ("vff", r"VFF, $\alpha=0.02$", "lt_vff", "#8E6BB8", "^", 8.0), ("mff", "MFF", "lt_mffmix_t", "#C2185B", "o", 7.0)]
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

    def val(e, r, k):
        return Lv[(Lv.estimator == e) & (Lv.regime == r) & (Lv.strategy == k)].loss.iloc[0]
    verbose_print("CHECK MFF - VFF vs verified (2 dp): max |diff| = %.3f" % max(abs(round(val(e, r, "mff") - val(e, r, "vff"), 2) - ref[i])
                                                                   for e, ref in REF.items() for i, (r, _) in enumerate(REG)))
    span = max(max(val(e, r, k) for e, _ in EST for k, *_ in ST) - min(val(e, r, k) for e, _ in EST for k, *_ in ST) for r, _ in REG) * 1.12
    with plt.rc_context(RC):
        fig, axs = plt.subplots(3, 1, figsize=(W, 4.6))
        for ax, (rk, rl) in zip(axs, REG):
            dot_panel(ax, lambda e, k: val(e, rk, k), ST, span, rl)
        fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.012)
        dot_legend(fig, ST)
        fig.subplots_adjust(left=0.19, right=0.975, top=0.9, bottom=0.1, hspace=0.62)
        fig.savefig(image("ch6_practical_appendix_vix.png"))
        plt.close(fig)
    verbose_print(Lv.pivot_table(index=["estimator", "strategy"], columns="regime", values="loss")[["calm", "mid", "stress"]].round(2).to_string())
    verbose_print("pair-days per condition:", ndays, f"| common span {span:.1f}")


if __name__ == "__main__":
    main()
