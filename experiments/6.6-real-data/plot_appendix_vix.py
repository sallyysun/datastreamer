"""Appendix figure: each strategy's own mean log prediction loss by VIX condition (t4 copula), in the design of the main
change-day loss figure (plot_change_days_levels.py). Strategies: fixed ESS 99 / 199 / 399, VFF, MFF (mixture density).
Conditions from the VIX close, forward-filled to each pair-day: calm VIX < 20, moderate 20 <= VIX < 25, stress VIX >= 25.
Mean = mean over pairs of each pair's mean loss over its days in the condition, x1e-3 nats/day/pair. Panels share one span.
Checks MFF - VFF against the verified differences. Writes vix_levels.csv and images/ch6_practical_appendix_vix.png.
Usage: python3 plot_appendix_vix.py <scratch dir> <vix.csv>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); S, VX = sys.argv[1], sys.argv[2]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
ST = [("fixed99", r"Fixed, $N_{\mathrm{eff}}=99$", "lt_fixed_99", "#8a8a8a", "D", 7.5), ("fixed199", r"Fixed, $N_{\mathrm{eff}}=199$", "lt_fixed_199", "#6E6E6E", "s", 5.5),
      ("fixed399", r"Fixed, $N_{\mathrm{eff}}=399$", "lt_fixed_399", "#333333", "p", 6.5), ("vff", r"VFF, $\alpha=0.02$", "lt_vff", "#8E6BB8", "^", 8.0),
      ("mff", "MFF", "lt_mffmix_t", "#C2185B", "o", 7.0)]
REG = [("calm", "Calm (VIX < 20)"), ("mid", "Moderate (20 ≤ VIX < 25)"), ("stress", "Stress (VIX ≥ 25)")]
vix = pd.read_csv(VX, index_col=0).iloc[:, 0]; vix.index = pd.to_datetime(vix.index)
rows, ndays = [], {}
for e, d in EST:
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        p = os.path.basename(f)[:-4]; x = pd.read_csv(f, parse_dates=["date"]).iloc[560:]; x = x[x.date >= "2008-01-01"]
        v = vix.reindex(pd.DatetimeIndex(x.date)).ffill().values
        assert not np.isnan(v).any()
        mask = {"calm": v < 20, "mid": (v >= 20) & (v < 25), "stress": v >= 25}
        for rk, m in mask.items():
            if e == "RMPL": ndays[rk] = ndays.get(rk, 0) + int(m.sum())
            for k, _, c, *_ in ST: rows.append(dict(estimator=e, pair=p, regime=rk, strategy=k, loss=-1e3 * x[c].values[m].mean()))
P = pd.DataFrame(rows); Lv = P.groupby(["estimator", "regime", "strategy"]).loss.mean().reset_index()
Lv.to_csv(os.path.join(HERE, "results", "vix_levels.csv"), index=False)
val = lambda e, r, k: Lv[(Lv.estimator == e) & (Lv.regime == r) & (Lv.strategy == k)].loss.iloc[0]
REF = {"RMPL": (-1.79, -2.65, -8.67), "Counting matrix": (-2.40, -1.56, -5.94), "Hermite series": (-3.64, -2.45, -5.75)}
print("CHECK MFF - VFF vs verified (2 dp): max |diff| = %.3f" % max(abs(round(val(e, r, "mff") - val(e, r, "vff"), 2) - ref[i])
                                                               for e, ref in REF.items() for i, (r, _) in enumerate(REG)))
span = max(max(val(e, r, k) for e, _ in EST for k, *_ in ST) - min(val(e, r, k) for e, _ in EST for k, *_ in ST) for r, _ in REG) * 1.12
with plt.rc_context(RC):
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.6))
    for ax, (rk, rl) in zip(axs, REG):
        allv = [val(e, rk, k) for e, _ in EST for k, *_ in ST]; mid = 0.5 * (max(allv) + min(allv))
        for ei, (e, _) in enumerate(EST):
            xs = [val(e, rk, k) for k, *_ in ST]
            ax.plot([min(xs), max(xs)], [ei, ei], color="#cfcec9", lw=1.6, zorder=2, solid_capstyle="round")
            for k, lab, c, col, mk, ms in ST:
                if k == "fixed99":   # hollow and on top, as in the main loss figure
                    ax.plot(val(e, rk, k), ei, marker=mk, ms=ms, mfc="none", mec=col, mew=1.1, ls="none", zorder=6)
                else:
                    ax.plot(val(e, rk, k), ei, marker=mk, ms=ms, color=col, mec="white", mew=0.6, ls="none", zorder=3 if k.startswith("fixed") else 4)
        ax.set_xlim(mid - span / 2, mid + span / 2); ax.set_ylim(len(EST) - 0.45, -0.55); ax.set_title(rl)
        ax.set_yticks(range(len(EST))); ax.set_yticklabels([e for e, _ in EST]); ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for ei in range(1, len(EST)): ax.axhline(ei - 0.5, color="#ecebe7", lw=0.6, zorder=1)
    fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.012)
    fig.legend([Line2D([], [], marker=mk, ls="none", ms=ms, mfc="none", mec=c, mew=1.1) if k == "fixed99" else
                Line2D([], [], marker=mk, ls="none", ms=ms, color=c, mec="white") for k, _, _, c, mk, ms in ST], [l for _, l, *_ in ST],
               loc="upper center", ncol=len(ST), frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.19, right=0.975, top=0.9, bottom=0.1, hspace=0.62)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_appendix_vix.png"))
    plt.close(fig)
print(Lv.pivot_table(index=["estimator", "strategy"], columns="regime", values="loss")[["calm", "mid", "stress"]].round(2).to_string())
print("pair-days per condition:", ndays, f"| common span {span:.1f}")
