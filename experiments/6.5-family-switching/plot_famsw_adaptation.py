"""Plot time to identify the new copula family after a switch."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402
from plot_famsw_initial_id import boxes  # noqa: E402

plt = ce.plt
SW = [(g, f"{C.FAMILY_MATH[a]} → {C.FAMILY_MATH[b]}", b, C.FAMILY_COLOUR[b]) for g, a, b in C.SWITCHES]
MINUS = "−"


def load():
    D = {g: pd.read_csv(ce.R + f"famsw_ident_{g}.csv") for g, _, _, _ in SW}
    for g, _, tgt, _ in SW:
        d = D[g]
        assert len(d) == 60 and (d.t50 >= 0).all() and (d.t90 >= 0).all() and (d.kind == "switch").all() and (d.target == tgt).all()
    L = pd.read_csv(ce.R + "famsw_rmpl_dl.csv")
    L = L[L.window == "rec_0_100"].set_index("regime")
    assert (L.loc[[g for g, _, _, _ in SW], "n"] == 60).all()
    return D, L


def reduction(ax, L, labels):
    for i, (g, _, _, c) in enumerate(SW[::-1]):
        r = L.loc[g]
        ax.barh(i + 1, r.dL, height=0.55, color=c, alpha=0.75, edgecolor="#444444", lw=0.7, zorder=3)
        ax.errorbar(r.dL, i + 1, xerr=r.dL_2se, color="#3a3a3a", lw=0.8, capsize=2, capthick=0.8, zorder=4)
        lo, hi = r.dL - r.dL_2se, r.dL + r.dL_2se
        ax.text(hi + 1.5 if r.dL >= 0 else lo - 1.5, i + 1, f"{r.dL:+.1f} ± {r.dL_2se:.1f}".replace("-", MINUS), va="center",
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


def figure(D, L):
    rev = SW[::-1]
    labels = [l for _, l, _, _ in rev]
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(ce.W, 3.7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.8, wspace=0.12, left=0.19, right=0.97, top=0.93, bottom=0.11)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
        ax_c = fig.add_subplot(gs[1, :])
        for ax, col, title in zip([ax_a, ax_b], ["t50", "t90"], ["(a) 50% weight on the new family", "(b) 90% weight on the new family"]):
            boxes(ax, [D[g][col].values for g, _, _, _ in rev], [c for _, _, _, c in rev], labels)
            ax.set_title(title, fontsize=8, pad=3, loc="left")
        plt.setp(ax_b.get_yticklabels(), visible=False)
        fig.text(0.58, 0.505, "Observations after the switch until the threshold is first reached", ha="center", fontsize=7.5)
        reduction(ax_c, L, labels)
        fig.savefig("images/ch6_famswitch_adaptation.png")
        plt.close(fig)


def main():
    D, L = load()
    figure(D, L)
    for g, l, _, _ in SW:
        print(f"{l:22s} t50 med {np.median(D[g].t50):g}  t90 med {np.median(D[g].t90):g}  dL100 {L.loc[g].dL:+.1f} ± {L.loc[g].dL_2se:.1f}")


if __name__ == "__main__":
    main()
