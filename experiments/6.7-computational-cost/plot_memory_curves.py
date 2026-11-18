"""Plot test RMSE against retained state at validation-selected gains."""
import sys
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
FAMS = [(f, C.FAMILY_LONG[f]) for f in C.FAMILIES_T31]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
EST = [("mpl_dskll", "RMPL (KLL $k$)", ce.COL["mle"], "^"), ("xiao", "Counting Matrix (bins)", ce.COL["xiao"], "o"),
       ("hermite", "Hermite Series (order)", ce.COL["hermite"], "s")]


def frontier(val, SD, f, s, e):
    """(retained KiB, mean test RMSE x 10^-2) per approximation size, at its validation-best gain."""
    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e)]
    return [(r.peak_bytes / 1024, 100 * SD.loc[(f, s, e, r.approx, r.gain)].mean()) for _, r in v.loc[v.groupby("approx").rmse.idxmin()].iterrows()]


def panel(ax, fr):
    for e, lab, col, mk in EST:
        ax.plot([p[0] for p in fr[e]], [p[1] for p in fr[e]], color=col, marker=mk, ms=3.0, lw=1.1, label=lab)
    best = [min(p[1] for p in fr[e]) for e, *_ in EST]
    top = max(p[1] for p in fr["xiao"])
    lo = min(best)
    top = max(top, max(best) + 0.25 * (top - lo))
    ax.set_ylim(lo - 0.08 * (top - lo), top + 0.3 * (top - lo))
    for b in C.BUDGETS:
        ax.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 8, 32, 128])
    ax.set_xticklabels(["2", "8", "32", "128"])
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)


def main():
    sw = pd.read_csv("results/budget_sweep_n15k.csv")
    SD = pd.read_csv("results/budget_seeds_n15k.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
    val = sw[sw.split == "val"]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(4, 3, figsize=(ce.W, 6.4), sharex=True)
        for i, (f, fl) in enumerate(FAMS):
            for j, (s, sl) in enumerate(SC):
                panel(axs[i, j], {e: frontier(val, SD, f, s, e) for e, *_ in EST})
                if i == 0:
                    axs[i, j].set_title(sl, fontsize=8, pad=3, loc="left")
                if j == 0:
                    axs[i, j].set_ylabel(fl, fontsize=7.5)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=1.6, columnspacing=1.2)
        fig.text(0.53, 0.01, "Retained state (KiB, log scale); dotted lines mark the 16, 64 and 256 KiB budgets", ha="center", fontsize=7.5)
        fig.text(0.005, 0.5, "Spearman RMSE ($\\times10^{-2}$)", rotation=90, va="center", fontsize=7.5)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.07, hspace=0.18, wspace=0.22)
        fig.savefig("images/ch6_budget_memory_curves_n15k.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
