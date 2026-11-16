"""Section 6.7 processing-time figure: time per observation (update + estimate, one thread, gain 0.02, mean of 5 runs,
n = 15,000 streams; results/budget_time_table_n15k.csv from prepare_data.py) against retained state, for every tested
approximation size of each estimator.
Usage: python3 plot_processing_time.py -> images/ch6_budget_processing_time.png"""
import sys
import pandas as pd

sys.path.insert(0, "../common")
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
E = [("mpl_dskll", "RMPL", ce.COL["mle"], "^"), ("xiao", "Counting Matrix", ce.COL["xiao"], "o"),
     ("hermite", "Hermite Series", ce.COL["hermite"], "s")]

for tfile, outname in [("results/budget_time_table_n15k.csv", "images/ch6_budget_processing_time.png")]:
    T = pd.read_csv(tfile)
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(core.W * 0.62, 2.5))
        for e, lab, c, m in E:
            x = T[T.estimator == e]
            ax.plot(x.kib, x.us, color=c, marker=m, ms=3, lw=1.1, label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
        ax.set_yticks([0.25, 1, 4, 16, 64]); ax.set_yticklabels(["0.25", "1", "4", "16", "64"])
        for b in (16, 64, 256): ax.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
        ax.grid(True, color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("Retained state (KiB)", fontsize=7.5); ax.set_ylabel("Time per observation (µs)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left")
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.17)
        fig.savefig(outname); plt.close(fig)
