"""Mini box plot replacing the Hedge-versus-equal table (t4 copula, full evaluation period). For each pair: mean over its
evaluation days of (Hedge MFF loss - equal-weight mixture loss), equal mixture = -log((1/5) sum_k exp(lt_fixed_k)).
Boxes (estimator colours; short two-entry legend; symbols explained in the caption): the 32 per-pair means. Diamond: mean over pairs (the table value). Black bar: its 95% moving-block bootstrap
interval from hedge_change_results.csv (analyze_hedge_change.py). Checks the recomputed means against that file.
Writes images/ch6_practical_hedge_equal_mini.png. Usage: python3 plot_appendix_hedge_mini.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
FIX = [39, 99, 199, 399, 1999]
ECOL = {"RMPL": "#1baf7a", "Counting matrix": "#2a78d6", "Hermite series": "#eb6834"}   # current Chapter 6 estimator colours (COL in experiments/plot_constant_ess.py)
CI = pd.read_csv(os.path.join(HERE, "results", "hedge_change_results.csv")); CI = CI[(CI.analysis == "hedge-equal") & (CI.days == "full")].set_index("estimator")
vals = {}
for e, d in EST:
    v = []
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        x = pd.read_csv(f, parse_dates=["date"]).iloc[560:]; x = x[x.date >= "2008-01-01"]
        eq = -np.log(np.mean(np.exp(np.stack([x[f"lt_fixed_{k}"].values for k in FIX])), axis=0))
        v.append(1e3 * np.mean(-x.lt_mffmix_t.values - eq))
    vals[e] = np.array(v)
print("CHECK mean over pairs vs hedge_change_results.csv: max |diff| = %.2e" % max(abs(vals[e].mean() - CI.loc[e, "est"]) for e, _ in EST))
with plt.rc_context(RC):
    fig, ax = plt.subplots(figsize=(W, 1.9))
    for i, (e, _) in enumerate(EST):
        ax.boxplot(vals[e], positions=[i], orientation="horizontal", widths=0.5, patch_artist=True,
                   medianprops=dict(color="#111111", lw=2.0), boxprops=dict(facecolor=ECOL[e], edgecolor="#555555", lw=0.9, alpha=0.85),
                   whiskerprops=dict(color="#555555", lw=0.9), capprops=dict(color="#555555", lw=0.9),
                   flierprops=dict(marker="o", ms=3.2, mfc="none", mec="#555555", mew=0.8))
        ax.plot([CI.loc[e, "lo"], CI.loc[e, "hi"]], [i - 0.36, i - 0.36], color="#111111", lw=1.6, solid_capstyle="butt", zorder=5)
        ax.plot(CI.loc[e, "est"], i - 0.36, marker="D", ms=4.2, mfc="white", mec="#111111", mew=0.8, ls="none", zorder=6)
    ax.axvline(0, color="#8a8984", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(EST))); ax.set_yticklabels([e for e, _ in EST]); ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(EST) - 0.55, -0.65); ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel(r"Mean prediction-loss difference: Hedge $-$ Equal ($\times10^{-3}$ nats/day)", fontsize=8)
    h = [Patch(facecolor="#b3b3b3", edgecolor="#555555", lw=0.9), Line2D([], [], color="#111111", lw=1.6, marker="D", ms=4.2, mfc="white", mec="#111111")]
    fig.legend(h, ["Spread across the 32 pairs", "Mean with 95% interval"], loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.58, 1.0), columnspacing=2.0, handlelength=1.6)
    fig.subplots_adjust(left=0.19, right=0.975, top=0.85, bottom=0.25)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_hedge_equal_mini.png"))
    plt.close(fig)
for e, _ in EST:
    v = vals[e]; print(f"{e:16s} mean {v.mean():+.2f} [{CI.loc[e,'lo']:+.2f}, {CI.loc[e,'hi']:+.2f}] median {np.median(v):+.2f} "
                       f"IQR [{np.percentile(v,25):+.2f}, {np.percentile(v,75):+.2f}] range [{v.min():+.2f}, {v.max():+.2f}] pairs Hedge better {int((v < 0).sum())}/32")
