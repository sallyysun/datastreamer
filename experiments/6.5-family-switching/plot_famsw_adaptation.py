"""Section 6.5 post-switch adaptation, RMPL / KLL configuration, 60 replications per switch. (a), (b): per-seed first-crossing
times after the switch at n = 2500 until the new family's weight (summed over the five memories) reaches 50% / 90% of the
family-augmented MFF weight (results/famsw_ident_<switch>.csv, kind = switch), box plots as ch6_famswitch_switch_id.png.
(c): reduction in excess log loss, L(Gaussian-only MFF) - L(family-augmented MFF), over observations 2500-2599, x1e-3
nats/obs, mean ± 2 SE over paired seeds (results/famsw_rmpl_dl.csv, window rec_0_100, from famsw_rmpl.py).
Usage: python3 plot_famsw_adaptation.py -> images/ch6_famswitch_adaptation.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
SW = [(g, f"{C.FAMILY_MATH[a]} → {C.FAMILY_MATH[b]}", b, C.FAMILY_COLOUR[b]) for g, a, b in C.SWITCHES]
D = {g: pd.read_csv(core.R + f"famsw_ident_{g}.csv") for g, _, _, _ in SW}
for g, _, tgt, _ in SW:
    d = D[g]
    assert len(d) == 60 and (d.t50 >= 0).all() and (d.t90 >= 0).all() and (d.kind == "switch").all() and (d.target == tgt).all()
L = pd.read_csv(core.R + "famsw_rmpl_dl.csv")
L = L[L.window == "rec_0_100"].set_index("regime")
assert (L.loc[[g for g, _, _, _ in SW], "n"] == 60).all()
labels = [l for _, l, _, _ in SW][::-1]

with plt.rc_context(ce.RC):
    fig = plt.figure(figsize=(core.W, 3.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.8, wspace=0.12, left=0.19, right=0.97, top=0.93, bottom=0.11)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a); ax_c = fig.add_subplot(gs[1, :])
    for ax, col, title in zip([ax_a, ax_b], ["t50", "t90"], ["(a) 50% weight on the new family", "(b) 90% weight on the new family"]):
        data = [D[g][col].values for g, _, _, _ in SW][::-1]
        bp = ax.boxplot(data, orientation="horizontal", whis=1.5, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                        capprops=dict(color="#555555", lw=0.8),
                        flierprops=dict(marker="o", ms=2.6, mfc="none", mec="#555555", mew=0.6))
        for patch, (_, _, _, c) in zip(bp["boxes"], SW[::-1]):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#444444"); patch.set_linewidth(0.7)
        ax.set_yticks(range(1, 4)); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, 1500); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xticks([250, 750, 1250], minor=True)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", which="both", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, pad=3, loc="left")
    plt.setp(ax_b.get_yticklabels(), visible=False)
    fig.text(0.58, 0.505, "Observations after the switch until the threshold is first reached", ha="center", fontsize=7.5)

    for i, (g, _, _, c) in enumerate(SW[::-1]):
        r = L.loc[g]
        ax_c.barh(i + 1, r.dL, height=0.55, color=c, alpha=0.75, edgecolor="#444444", lw=0.7, zorder=3)
        ax_c.errorbar(r.dL, i + 1, xerr=r.dL_2se, color="#3a3a3a", lw=0.8, capsize=2, capthick=0.8, zorder=4)
        lo, hi = r.dL - r.dL_2se, r.dL + r.dL_2se
        ax_c.text(hi + 1.5 if r.dL >= 0 else lo - 1.5, i + 1, f"{r.dL:+.1f} ± {r.dL_2se:.1f}".replace("-", "−"), va="center",
                  ha="left" if r.dL >= 0 else "right", fontsize=6.5, color="#333333")
    ax_c.axvline(0, color="#222222", lw=0.7, zorder=2)
    ax_c.set_yticks(range(1, 4)); ax_c.set_yticklabels(labels, fontsize=7.5); ax_c.tick_params(axis="y", length=0)
    ax_c.set_xlim(-65, 45); ax_c.set_ylim(0.4, 3.6)
    ax_c.grid(True, axis="x", color=ce.GRID, lw=0.5, zorder=0); ax_c.set_axisbelow(True)
    ax_c.set_title("(c) Reduction in excess log loss, first 100 observations after the switch", fontsize=8, pad=3, loc="left")
    ax_c.set_xlabel("Reduction relative to Gaussian-only MFF ($\\times10^{-3}$ nats per observation)", fontsize=7.5)
    fig.savefig("images/ch6_famswitch_adaptation.png"); plt.close(fig)

for g, l, _, _ in SW:
    print(f"{l:22s} t50 med {np.median(D[g].t50):g}  t90 med {np.median(D[g].t90):g}  dL100 {L.loc[g].dL:+.1f} ± {L.loc[g].dL_2se:.1f}")
print("wrote images/ch6_famswitch_adaptation.png")
