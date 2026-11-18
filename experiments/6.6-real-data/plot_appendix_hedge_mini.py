"""Section 6.6 appendix: per-pair mean of Hedge MFF loss minus the exact equal-weight mixture of the five arms (t4
copula, full evaluation period), box plots over the 32 pairs with the pair mean and its bootstrap interval from
results/hedge_change_results.csv (checked against the recomputed means).
Usage: python3 plot_appendix_hedge_mini.py <scratch dir> -> images/ch6_practical_hedge_equal_mini.png"""
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from real_common import ECOL, EST, FIX, GRID, RC, W, box_style, eval_days, image, outputs, plt, read, result


def differences(S):
    vals = {}
    for e, d in EST:
        v = []
        for f in outputs(S, d).values():
            x = eval_days(read(f))
            eq = -np.log(np.mean(np.exp(np.stack([x[f"lt_fixed_{k}"].values for k in FIX])), axis=0))
            v.append(1e3 * np.mean(-x.lt_mffmix_t.values - eq))
        vals[e] = np.array(v)
    return vals


def figure(vals, CI):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(W, 1.9))
        for i, (e, _) in enumerate(EST):
            ax.boxplot(vals[e], positions=[i], orientation="horizontal", widths=0.5, patch_artist=True, **box_style(ECOL[e], 0.85))
            ax.plot([CI.loc[e, "lo"], CI.loc[e, "hi"]], [i - 0.36, i - 0.36], color="#111111", lw=1.6, solid_capstyle="butt", zorder=5)
            ax.plot(CI.loc[e, "est"], i - 0.36, marker="D", ms=4.2, mfc="white", mec="#111111", mew=0.8, ls="none", zorder=6)
        ax.axvline(0, color="#8a8984", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_yticks(range(len(EST)))
        ax.set_yticklabels([e for e, _ in EST])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(EST) - 0.55, -0.65)
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"Mean prediction-loss difference: Hedge $-$ Equal ($\times10^{-3}$ nats/day)", fontsize=8)
        h = [Patch(facecolor="#b3b3b3", edgecolor="#555555", lw=0.9), Line2D([], [], color="#111111", lw=1.6, marker="D", ms=4.2, mfc="white", mec="#111111")]
        fig.legend(h, ["Spread across the 32 pairs", "Mean with 95% interval"], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.58, 1.0), columnspacing=2.0, handlelength=1.6)
        fig.subplots_adjust(left=0.19, right=0.975, top=0.85, bottom=0.25)
        fig.savefig(image("ch6_practical_hedge_equal_mini.png"))
        plt.close(fig)


def main():
    CI = pd.read_csv(result("hedge_change_results.csv"))
    CI = CI[(CI.analysis == "hedge-equal") & (CI.days == "full")].set_index("estimator")
    vals = differences(sys.argv[1])
    print("CHECK mean over pairs vs hedge_change_results.csv: max |diff| = %.2e" % max(abs(vals[e].mean() - CI.loc[e, "est"]) for e, _ in EST))
    figure(vals, CI)
    for e, _ in EST:
        v = vals[e]
        print(f"{e:16s} mean {v.mean():+.2f} [{CI.loc[e,'lo']:+.2f}, {CI.loc[e,'hi']:+.2f}] median {np.median(v):+.2f} "
              f"IQR [{np.percentile(v,25):+.2f}, {np.percentile(v,75):+.2f}] range [{v.min():+.2f}, {v.max():+.2f}] pairs Hedge better {int((v < 0).sum())}/32")


if __name__ == "__main__":
    main()
