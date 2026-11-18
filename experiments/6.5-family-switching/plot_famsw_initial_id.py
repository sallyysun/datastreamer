"""Plot time to identify the generating copula family."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
FAMS = [(f, C.FAMILY_LONG[f], C.FAMILY_COLOUR[f]) for f in C.FAMILIES_T31]


def load():
    D = {f: pd.read_csv(ce.R + f"famsw_ident_{f}.csv") for f, _, _ in FAMS}
    for d in D.values():
        assert len(d) == 60 and (d.t50 >= 0).all() and (d.t90 >= 0).all() and (d.kind == "initial").all()
    return D


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


def figure(D):
    rev = FAMS[::-1]   # Gaussian at the top
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 2, figsize=(ce.W, 2.3), sharey=True)
        for ax, col, title in zip(axs, ["t50", "t90"], ["(a) 50% weight", "(b) 90% weight"]):
            boxes(ax, [D[f][col].values for f, _, _ in rev], [c for _, _, c in rev], [l for _, l, _ in rev])
            ax.set_title(title, fontsize=8, pad=3, loc="left")
        fig.text(0.56, 0.02, "Observations until the true family first reaches the threshold", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.2, wspace=0.12)
        fig.savefig("images/ch6_famswitch_initial_id.png")
        plt.close(fig)


def main():
    D = load()
    figure(D)
    for f, l, _ in FAMS:
        def q(x):
            return f"{np.percentile(x, 50):g} [{np.percentile(x, 25):g}, {np.percentile(x, 75):g}]"
        print(f"{l:16s} t50 {q(D[f].t50):22s} t90 {q(D[f].t90)}")


if __name__ == "__main__":
    main()
