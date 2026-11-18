"""Section 6.7 retained state of the validation-selected configurations at each storage budget: the maximum over the 12
family x scenario selections of the peak store_bytes() (KiB), from results/budget_selected.csv.
Writes images/ch6_budget_storage_n15k.png."""
import sys
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [("mpl_dskll", "RMPL", ce.COL["mle"]), ("xiao", "Counting Matrix", ce.COL["xiao"]), ("hermite", "Hermite Series", ce.COL["hermite"])]
TICKS = [2, 4, 8, 16, 32, 64, 128, 256]


def figure(sel):
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.72, 2.5))
        for i, b in enumerate(C.BUDGETS):
            ax.hlines(b, i - 0.45, i + 0.45, color="#444444", lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.text(i - 0.45, b * 1.07, "budget", va="bottom", ha="left", fontsize=6, color="#444444")
            for j, (e, lab, col) in enumerate(EST):
                top = sel[(sel.estimator == e) & (sel.budget_kib == b)].kib.max()
                xp = i + (j - 1) * bw
                ax.bar(xp, top, width=bw * 0.9, color=col, alpha=0.85, zorder=3, label=lab if i == 0 else None)
                ax.text(xp, top * 0.9, f"{top:.1f}", ha="center", va="top", fontsize=5.6, color="white")
        ax.set_yscale("log", base=2)
        ax.set_yticks(TICKS)
        ax.set_yticklabels([str(t) for t in TICKS])
        ax.set_ylim(2, 400)
        ax.set_xlim(-0.55, 2.75)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{b} KiB" for b in C.BUDGETS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Storage budget", fontsize=7.5)
        ax.set_ylabel("Maximum selected\nretained state (KiB)", fontsize=7.5)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.53, 1.0), handlelength=1.2, columnspacing=1.0)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.17)
        fig.savefig("images/ch6_budget_storage_n15k.png")
        plt.close(fig)


def main():
    sel = pd.read_csv("results/budget_selected.csv")
    sel["kib"] = sel.peak_val / 1024
    figure(sel)
    print(sel.groupby(["estimator", "budget_kib"]).kib.agg(["min", "median", "max"]).round(2))
    print("wrote images/ch6_budget_storage_n15k.png")


if __name__ == "__main__":
    main()
