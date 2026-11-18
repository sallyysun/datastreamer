"""Plot family-switching timing, weights, and predictive accuracy."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402
from report_output import verbose_print  # noqa: E402

plt = ce.plt
RESULTS = Path(ce.R)
IMAGES = Path(ce.OUT)
INITIAL = [(f, C.FAMILY_LONG[f], C.FAMILY_COLOUR[f]) for f in C.FAMILIES_T31]
SWITCHES = [(g, f"{C.FAMILY_MATH[a]} → {C.FAMILY_MATH[b]}", C.FAMILY_COLOUR[b]) for g, a, b in C.SWITCHES]
SWITCH_TARGETS = {g: b for g, _, b in C.SWITCHES}
WEIGHT_FAMILIES = [(f, C.FAMILY_MATH[f], C.FAMILY_COLOUR[f]) for f in C.FAMILIES_T31]
ESTIMATORS = [(e, C.EST_LABEL[e], C.EST_COLOUR_KEY[e]) for e in C.ESTIMATORS]
ACCURACY_METRICS = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]


def load_identification(regimes, kind):
    data = {}
    for regime, _, _ in regimes:
        rows = pd.read_csv(RESULTS / f"famsw_ident_{regime}.csv")
        assert len(rows) == len(C.SEEDS) and (rows.kind == kind).all()
        assert (rows.t50 >= 0).all() and (rows.t90 >= 0).all()
        if kind == "switch":
            assert (rows.target == SWITCH_TARGETS[regime]).all()
        data[regime] = rows
    return data


def boxes(ax, data, colours, labels):
    bp = ax.boxplot(data, orientation="horizontal", whis=1.5, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8),
                    flierprops=dict(marker="o", ms=2.6, mfc="none", mec="#555555", mew=0.6))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.7)
    ax.set_yticks(range(1, len(data) + 1))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xticks([250, 750, 1250], minor=True)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", which="both", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def threshold_panels(axes, data, regimes, titles):
    ordered = regimes[::-1]
    labels = [label for _, label, _ in ordered]
    colours = [colour for _, _, colour in ordered]
    for ax, column, title in zip(axes, ("t50", "t90"), titles):
        boxes(ax, [data[regime][column].values for regime, _, _ in ordered], colours, labels)
        ax.set_title(title, fontsize=8, pad=3, loc="left")


def plot_identification():
    data = load_identification(INITIAL, "initial")
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 2, figsize=(ce.W, 2.3), sharey=True)
        threshold_panels(axs, data, INITIAL, ("(a) 50% weight", "(b) 90% weight"))
        fig.text(0.56, 0.02, "Observations until the true family first reaches the threshold", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.2, wspace=0.12)
        fig.savefig(IMAGES / "ch6_famswitch_initial_id.png")
        plt.close(fig)
    if os.environ.get("VERBOSE") == "1":
        def quartiles(values):
            return f"{np.percentile(values, 50):g} [{np.percentile(values, 25):g}, {np.percentile(values, 75):g}]"
        for regime, label, _ in INITIAL:
            rows = data[regime]
            verbose_print(f"{label:16s} t50 {quartiles(rows.t50):22s} t90 {quartiles(rows.t90)}")


def reduction_panel(ax, L, labels):
    for i, (g, _, c) in enumerate(SWITCHES[::-1]):
        r = L.loc[g]
        ax.barh(i + 1, r.dL, height=0.55, color=c, alpha=0.75, edgecolor="#444444", lw=0.7, zorder=3)
        ax.errorbar(r.dL, i + 1, xerr=r.dL_2se, color="#3a3a3a", lw=0.8, capsize=2, capthick=0.8, zorder=4)
        lo, hi = r.dL - r.dL_2se, r.dL + r.dL_2se
        ax.text(hi + 1.5 if r.dL >= 0 else lo - 1.5, i + 1, f"{r.dL:+.1f} ± {r.dL_2se:.1f}".replace("-", "−"), va="center",
                ha="left" if r.dL >= 0 else "right", fontsize=6.5, color="#333333")
    ax.axvline(0, color="#222222", lw=0.7, zorder=2)
    ax.set_yticks(range(1, 4))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-65, 45)
    ax.set_ylim(0.4, 3.6)
    ax.grid(True, axis="x", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(c) Reduction in excess log loss, first 100 observations after the switch", fontsize=8, pad=3, loc="left")
    ax.set_xlabel("Reduction relative to Gaussian-only MFF ($\\times10^{-3}$ nats per observation)", fontsize=7.5)


def plot_adaptation():
    data = load_identification(SWITCHES, "switch")
    losses = pd.read_csv(RESULTS / "famsw_rmpl_dl.csv")
    early = losses[losses.window == "rec_0_100"].set_index("regime")
    assert (early.loc[[regime for regime, _, _ in SWITCHES], "n"] == len(C.SEEDS)).all()
    labels = [label for _, label, _ in SWITCHES[::-1]]
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(ce.W, 3.7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.8, wspace=0.12, left=0.19, right=0.97, top=0.93, bottom=0.11)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
        ax_c = fig.add_subplot(gs[1, :])
        threshold_panels((ax_a, ax_b), data, SWITCHES,
                         ("(a) 50% weight on the new family", "(b) 90% weight on the new family"))
        plt.setp(ax_b.get_yticklabels(), visible=False)
        fig.text(0.58, 0.505, "Observations after the switch until the threshold is first reached", ha="center", fontsize=7.5)
        reduction_panel(ax_c, early, labels)
        fig.savefig(IMAGES / "ch6_famswitch_adaptation.png")
        plt.close(fig)
    if os.environ.get("VERBOSE") == "1":
        for regime, label, _ in SWITCHES:
            rows, loss = data[regime], early.loc[regime]
            verbose_print(f"{label:22s} t50 med {np.median(rows.t50):g}  t90 med {np.median(rows.t90):g}  "
                          f"dL100 {loss.dL:+.1f} ± {loss.dL_2se:.1f}")


def plot_weights():
    titles = [(g, f"{C.FAMILY_MATH[a]} $\\rightarrow$ {C.FAMILY_MATH[b]}") for g, a, b in C.SWITCHES]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.45), sharey=True)
        for ax, (g, title) in zip(axs, titles):
            W = pd.read_csv(RESULTS / f"famsw_path_{g}.csv").pivot_table(index="t", columns="family", values="mean")
            W = W[W.index >= 500][[f for f, _, _ in WEIGHT_FAMILIES]]
            W = W.div(W.sum(axis=1), axis=0)
            ax.stackplot(W.index, *[W[f] for f, _, _ in WEIGHT_FAMILIES], colors=[c for _, _, c in WEIGHT_FAMILIES], lw=0, zorder=2)
            ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
            ax.set_xlim(500, 5000)
            ax.set_xticks([1000, 2500, 4000])
            ax.set_ylim(0, 1)
            ax.set_title(title, fontsize=8, pad=3)
        axs[0].set_ylabel("Mean MFF weight", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in WEIGHT_FAMILIES]
        fig.legend(h, [l for _, l, _ in WEIGHT_FAMILIES], loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
                   handlelength=1.2, columnspacing=1.4)
        fig.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.18, wspace=0.08)
        fig.savefig(IMAGES / "ch6_famswitch_rmpl_weights.png")
        plt.close(fig)


def plot_accuracy():
    data = pd.read_csv(RESULTS / "famsw_accuracy.csv", float_precision="round_trip")
    post = data[data.window == "post"].set_index(["estimator", "regime", "metric"])
    switches = [(g, f"{C.FAMILY_MATH[a]} →\n{C.FAMILY_MATH[b]}") for g, a, b in C.SWITCHES]
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 2, figsize=(ce.W, 2.5), sharey=False)
        for ax, (m, ml), tag in zip(axs, ACCURACY_METRICS, ["(a)", "(b)"]):
            for j, (e, _, ck) in enumerate(ESTIMATORS):
                for i, (g, _) in enumerate(switches):
                    r = post.loc[(e, g, m)]
                    x = i + (j - 1) * bw
                    ax.bar(x, r.pct, width=bw * 0.92, color=ce.COL[ck], zorder=3)
                    ax.errorbar(x, r.pct, yerr=r.pct_2se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
            ax.axhline(0, color="#222222", lw=0.7, zorder=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels([label for _, label in switches], fontsize=7)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.55, 2.55)
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(f"{tag} {ml}", fontsize=8, pad=3, loc="left")
        axs[0].set_ylabel("Reduction relative to\nGaussian-only MFF (%)", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color=ce.COL[ck]) for _, _, ck in ESTIMATORS]
        fig.legend(h, [el for _, el, _ in ESTIMATORS], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
                   handlelength=1.2, columnspacing=1.4)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.8, bottom=0.18, wspace=0.18)
        fig.savefig(IMAGES / "ch6_famswitch_accuracy.png")
        plt.close(fig)


def main():
    plots = {
        "weights": plot_weights,
        "accuracy": plot_accuracy,
        "identification": plot_identification,
        "adaptation": plot_adaptation,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("all", *plots), default="all")
    mode = parser.parse_args().mode
    IMAGES.mkdir(exist_ok=True)
    if mode == "all":
        for plot in plots.values():
            plot()
    else:
        plots[mode]()


if __name__ == "__main__":
    main()
