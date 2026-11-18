"""Section 6.6 real-data examples (RMPL, t4-scored MFF): three pairs whose dependence behaves differently, with each
fixed memory's rho_S held before the day, MFF's, and MFF's weights (ch6_practical_examples); and one pair's rank
scatter in a calm and a break window (ch6_practical_example_scatter, appendix). Writes the window log-score totals to
results/practical_examples.txt. Usage: python3 plots_examples.py <scratch dir>"""
import sys

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from real_common import FIX, GRID, RC, W, image, plt, read, result, write_report
from plot_theme import STRATEGY, memory_colours

MEMORY = memory_colours(FIX)
FIXED_LINES = [(39, "#C8C8C8", "-", 0.7), (99, "#A8A8A8", (0, (5, 2)), 0.8), (199, "#888888", (0, (2, 1.5)), 0.8),
               (399, "#666666", (0, (6, 2, 1, 2)), 0.8), (1999, "#3A3A3A", "-", 0.8)]   # light = short memory
EXAMPLES = [("GLD_GDX", "2018-03-01", "2020-08-31", "Gold / gold miners (GLD/GDX)", "Stable, then an abrupt break"),
            ("ABT_JNJ", "2011-06-01", "2014-01-31", "Abbott / Johnson & Johnson", "Repeated swings"),
            ("PFE_MRK", "2024-01-01", "2026-11-11", "Pfizer / Merck", "Gradual rise and fall")]
SCATTER_PAIR = "GLD_GDX"
SCATTER_WINDOWS = [("2019-03-01", "2019-04-15", "Calm: 1 Mar to 15 Apr 2019"), ("2020-03-01", "2020-04-15", "Break: 1 Mar to 15 Apr 2020")]


def window(S, pair, first, last):
    x = read(f"{S}/practical_rmpl/out/{pair}.csv")
    return x[(x.date >= first) & (x.date <= last)].reset_index(drop=True)


def example_column(a, b, x, title, subtitle):
    d = x.date
    for z, (k, col, ls, lw) in enumerate(FIXED_LINES):
        a.plot(d, x[f"rho_fixed_{k}"], color=col, ls=ls, lw=lw, zorder=2 + z)
    a.plot(d, x.rho_mff_t, color=STRATEGY["mff"], lw=1.1, zorder=9)
    a.set_ylim(-0.1, 1.0)
    a.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    a.set_title(f"{title}\n{subtitle}", fontsize=7.4, linespacing=1.25)
    b.stackplot(d, *[x[f"wt_{k}"] for k in FIX], colors=[MEMORY[k] for k in FIX], lw=0)
    b.set_ylim(0, 1)
    b.set_yticks([0, 0.5, 1])
    for ax in (a, b):
        ax.set_xlim(d.iloc[0], d.iloc[-1])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.tick_params(labelbottom=False)
    b.tick_params(axis="x", labelsize=7)


def examples_figure(S, report):
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 3, figsize=(W, 3.7), sharey="row", gridspec_kw={"height_ratios": [1.5, 0.75]})
        for ci, (pair, first, last, title, subtitle) in enumerate(EXAMPLES):
            x = window(S, pair, first, last)
            example_column(axs[0, ci], axs[1, ci], x, title, subtitle)
            gains = ", ".join(f"{k} {x.lt_mffmix_t.sum() - x[f'lt_fixed_{k}'].sum():+.2f}" for k in FIX)
            report.append(f"{pair} {first}..{last} ({len(x)} days): MFF minus fixed (nats) {gains} | "
                          f"VFF {x.lt_mffmix_t.sum() - x.lt_vff.sum():+.2f}")
        axs[0, 0].set_ylabel(r"Estimated $\rho_S$")
        axs[1, 0].set_ylabel("MFF weight")
        lines = [Line2D([], [], color=col, ls=ls, lw=lw + 0.4) for _, col, ls, lw in FIXED_LINES] + [Line2D([], [], color=STRATEGY["mff"], lw=1.5)]
        fig.legend(lines, [f"Fixed {k}" for k, *_ in FIXED_LINES] + ["MFF"], loc="upper center", ncol=6, frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0.54, 1.0), handlelength=2.0, columnspacing=1.0,
                   title=r"Dependence estimates ($N_{\mathrm{eff}}$)", title_fontsize=6.8)
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=MEMORY[k]) for k in FIX], [str(k) for k in FIX], loc="lower center", ncol=5,
                   frameon=False, fontsize=6.8, bbox_to_anchor=(0.54, 0.0), handlelength=1.0, columnspacing=1.2,
                   title=r"MFF weight on candidate $N_{\mathrm{eff}}$", title_fontsize=6.8)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.2, hspace=0.12, wspace=0.08)
        fig.savefig(image("ch6_practical_examples.png"))
        plt.close(fig)


def scatter_figure(S, report):
    x = read(f"{S}/practical_rmpl/out/{SCATTER_PAIR}.csv")
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(W * 0.75, 2.45), sharex=True, sharey=True)
        for ax, (first, last, label) in zip(axs, SCATTER_WINDOWS):
            w = x[(x.date >= first) & (x.date <= last)]
            r = spearmanr(w.u, w.v).correlation
            ax.scatter(w.u, w.v, s=7, color="#3E4A56", alpha=0.75, lw=0)
            ax.plot([0, 1], [0, 1], color="#cfcfcf", lw=0.7, zorder=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_title(f"{label}\nsample $\\rho_S = {r:.2f}$ ({len(w)} days)", fontsize=7.4, linespacing=1.25)
            report.append(f"{SCATTER_PAIR} {label}: n {len(w)}, sample Spearman of (U, V) {r:.3f}")
        a, b = SCATTER_PAIR.split("_")
        axs[0].set_ylabel(f"{b} rank, $\\hat V$")
        fig.supxlabel(f"{a} rank, $\\hat U$", fontsize=8, y=0.03)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.17, wspace=0.12)
        fig.savefig(image("ch6_practical_example_scatter.png"))
        plt.close(fig)


def main():
    S = sys.argv[1]
    report = []
    examples_figure(S, report)
    scatter_figure(S, report)
    write_report(result("practical_examples.txt"), report)


if __name__ == "__main__":
    main()
