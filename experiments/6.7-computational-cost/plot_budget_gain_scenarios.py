"""Plot RMSE reductions at 64 and 256 KiB relative to 16 KiB."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [("mpl_dskll", "RMPL", ce.COL["mle"]), ("xiao", "Counting\nMatrix", ce.COL["xiao"]), ("hermite", "Hermite\nSeries", ce.COL["hermite"])]
SC = [("constant", "(a) Constant dependence"), ("drift", "(b) Gradual drift"), ("jump", "(c) Abrupt jump")]


def panel(ax, P, s, lo, hi):
    pos, data, cols, alphas = [], [], [], []
    for i, (e, _, c) in enumerate(EST):
        for j, b in enumerate((64, 256)):
            pos.append(i + (j - 0.5) * 0.36)
            cols.append(c)
            alphas.append(0.35 if b == 64 else 0.85)
            data.append(P[(P.scenario == s) & (P.estimator == e) & (P.budget == b)].pct.values)
    bp = ax.boxplot(data, positions=pos, widths=0.3, whis=1.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec="#555555", mew=0.6))
    for p, c, a in zip(bp["boxes"], cols, alphas):
        p.set_facecolor(c)
        p.set_alpha(a)
        p.set_edgecolor("#444444")
        p.set_linewidth(0.7)
    ax.axhline(0, color="#222222", lw=0.6, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l, _ in EST], fontsize=6.6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(np.floor(lo) - 0.3, np.ceil(hi) + 0.3)
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def figure(P, out):
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.4), sharey=True)
        for ax, (s, t) in zip(axs, SC):
            panel(ax, P, s, P.pct.min(), P.pct.max())
            ax.set_title(t, fontsize=7.8, pad=3, loc="left")
        axs[0].set_ylabel("RMSE reduction vs 16 KiB (%)", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.35), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.85)]
        fig.legend(h, ["64 KiB", "256 KiB"], loc="upper center", ncol=2, frameon=False, fontsize=6.6, bbox_to_anchor=(0.54, 1.0), handlelength=1.2)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.13, wspace=0.06)
        fig.savefig(out)
        plt.close(fig)


def summary(P):
    rows = []
    for (s, e, b), x in P.groupby(["scenario", "estimator", "budget"]):
        rows.append(dict(scenario=s, estimator=e, budget=b, median=x.pct.median(), min=x.pct.min(), max=x.pct.max(),
                         mc_se_median=x.pct_mc_se.median(), n_sig=int((x.pct > 2 * x.pct_mc_se).sum()),
                         n_neg_sig=int((x.pct < -2 * x.pct_mc_se).sum())))
    return pd.DataFrame(rows)


def main():
    P = pd.read_csv("results/budget_gain.csv")
    figure(P, "images/ch6_budget_accuracy_gain_scenarios.png")
    R = summary(P)
    R.to_csv("results/budget_gain_scenarios.csv", index=False)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
