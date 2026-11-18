"""Plot single-thread update and estimation time at matched storage budgets."""
import sys
import pandas as pd

sys.path.insert(0, "../common")
from budget_plot_style import budget_estimators, budget_guides, log2_axis  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = budget_estimators()


def main():
    T = pd.read_csv("results/budget_time_table_n15k.csv")
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.62, 2.5))
        for e, lab, c, m in EST:
            x = T[T.estimator == e]
            ax.plot(x.kib, x.us, color=c, marker=m, ms=3, lw=1.1, label=lab)
        log2_axis(ax.set_xscale, ax.set_xticks, ax.set_xticklabels, [1, 4, 16, 64, 256])
        log2_axis(ax.set_yscale, ax.set_yticks, ax.set_yticklabels, [0.25, 1, 4, 16, 64])
        budget_guides(ax)
        ax.grid(True, color=ce.GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Retained state (KiB)", fontsize=7.5)
        ax.set_ylabel("Time per observation (µs)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left")
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.17)
        fig.savefig("images/ch6_budget_processing_time.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
