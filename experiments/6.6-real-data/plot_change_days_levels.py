"""Mean predictive log loss on dependence-change days and other days (t4 copula; Chapter 6 style). Each method plotted on
its own (no differences): fixed ESS 199, VFF, MFF, for RMPL / counting matrix / Hermite. Change days: each pair's top 10%
of |rho_roll99(n) - rho_roll99(n-21)| (ex post classification; not used by any method). Mean = mean over pairs of each
pair's mean loss over its days in the set, x1e-3 nats/day/pair. Both panels span the same range of loss.
Writes change_days_levels.csv and images/ch6_practical_change_days_levels.png.
Usage: python3 plot_change_days_levels.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
FIXED = [int(k) for k in os.environ.get("FIXED", "99,199,399").split(",")]   # main: all three fixed memories
FIXSTY = {39: ("#A6A6A6", "D", 5.0), 99: ("#8a8a8a", "D", 7.5), 199: ("#6E6E6E", "s", 5.5), 399: ("#333333", "p", 6.5)}   # grey ramp as in the responsiveness figure
ST = [(f"fixed{k}", rf"Fixed, $N_{{\mathrm{{eff}}}}={k}$", f"lt_fixed_{k}", *FIXSTY[k]) for k in FIXED] + [("vff", r"VFF, $\alpha=0.02$", "lt_vff", "#8E6BB8", "^", 8.0), ("mff", "MFF", "lt_mffmix_t", "#C2185B", "o", 7.0)]
roll = {}
for f in sorted(glob.glob(f"{S}/practical_roll2/out/*.csv")):
    x = pd.read_csv(f, parse_dates=["date"]); x = x.assign(change=(x.rho_roll_99 - x.rho_roll_99.shift(21)).abs()).iloc[560:]
    roll[os.path.basename(f)[:-4]] = x[x.date >= "2008-01-01"].set_index("date")
pairs = list(roll); idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in roll.values()])))
CH = np.stack([roll[p].change.reindex(idx).values for p in pairs]); present = ~np.isnan(CH)
chg = (CH >= np.nanquantile(CH, 0.90, axis=1, keepdims=True)) & present
SETS = [("other", "Other days", (~chg) & present), ("change", "Large estimated dependence-change days (top 10% per pair)", chg)]
rows = []
for e, d in EST:
    P = {}
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        x = pd.read_csv(f, parse_dates=["date"]).set_index("date").iloc[560:]; P[os.path.basename(f)[:-4]] = x[x.index >= "2008-01-01"]
    for k, lab, c, *_ in ST:
        X = np.stack([-P[p][c].reindex(idx).values for p in pairs])
        for sk, _, M in SETS:
            rows.append(dict(estimator=e, strategy=k, days=sk, loss=1e3 * np.nanmean([np.nanmean(np.where(M[i], X[i], np.nan)) for i in range(len(pairs))])))
Lv = pd.DataFrame(rows); TAG = "" if FIXED == [99, 199, 399] else "_" + "_".join(map(str, FIXED))
Lv.to_csv(os.path.join(HERE, "results", f"change_days_levels{TAG}.csv"), index=False)
val = lambda e, k, s: Lv[(Lv.estimator == e) & (Lv.strategy == k) & (Lv.days == s)].loss.iloc[0]
span = max(max(val(e, k, s) for e, _ in EST for k, *_ in ST) - min(val(e, k, s) for e, _ in EST for k, *_ in ST) for s, *_ in SETS) * 1.15
with plt.rc_context(RC):
    fig, axs = plt.subplots(2, 1, figsize=(W, 3.4))
    for ax, (sk, sl, _) in zip(axs, SETS):
        xs_all = [val(e, k, sk) for e, _ in EST for k, *_ in ST]; mid = 0.5 * (max(xs_all) + min(xs_all))
        for ei, (e, _) in enumerate(EST):
            xs = [val(e, k, sk) for k, *_ in ST]
            ax.plot([min(xs), max(xs)], [ei, ei], color="#cfcec9", lw=1.6, zorder=2, solid_capstyle="round")
            for k, lab, c, col, mk, ms in ST:   # all on one line; VFF and MFF drawn above fixed forgetting
                if k == "fixed99":   # hollow and on top: it nearly coincides with MFF for RMPL and the counting matrix
                    ax.plot(val(e, k, sk), ei, marker=mk, ms=ms, mfc="none", mec=col, mew=1.1, ls="none", zorder=6)
                else:
                    ax.plot(val(e, k, sk), ei, marker=mk, ms=ms, color=col, mec="white", mew=0.6, ls="none", zorder=3 if k.startswith("fixed") else 4)
        ax.set_xlim(mid - span / 2, mid + span / 2); ax.set_ylim(len(EST) - 0.45, -0.55); ax.set_title(sl)
        ax.set_yticks(range(len(EST))); ax.set_yticklabels([e for e, _ in EST]); ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for ei in range(1, len(EST)): ax.axhline(ei - 0.5, color="#ecebe7", lw=0.6, zorder=1)
    fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.015)
    fig.legend([Line2D([], [], marker=mk, ls="none", ms=ms, mfc="none", mec=c, mew=1.1) if k == "fixed99" else
                Line2D([], [], marker=mk, ls="none", ms=ms, color=c, mec="white") for k, _, _, c, mk, ms in ST], [l for _, l, *_ in ST],
               loc="upper center", ncol=len(ST), frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.19, right=0.975, top=0.86, bottom=0.13, hspace=0.6)
    fig.savefig(os.path.join(HERE, "images", f"ch6_practical_final_change_days{TAG}.png"))
    plt.close(fig)
REF = {"RMPL": (-6.93, -1.61), "Counting matrix": (-6.36, -1.13), "Hermite series": (-3.92, -3.68)}   # mff_vs_fixed199.txt
print("CHECK MFF - fixed 199 level differences vs mff_vs_fixed199.txt (change, other): max |diff| = %.3f" % max(
    abs(val(e, "mff", s) - val(e, "fixed199", s) - r) for e, ref in REF.items() for s, r in zip(("change", "other"), ref)))
print(Lv.pivot_table(index=["estimator", "strategy"], columns="days", values="loss").round(2).to_string()); print(f"common span {span:.1f}")
