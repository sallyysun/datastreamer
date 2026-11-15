"""Section 6.5 post-switch family identification, RMPL / KLL configuration: box plots of the observations after the switch
at n = 2500 until the new family's weight (summed over the five memories N_eff 39-1999) first reaches 50% (a) and 90% (b)
of the family-augmented MFF weight, for the three switches, 60 replications each. Data: results/famsw_ident_<switch>.csv
(probe_strategies FAMSWITCH=1, kind = switch; t50/t90 are per-seed first-crossing times already counted from the switch;
all 60 reach both thresholds). Boxes coloured by the new family; style and axis as ch6_famswitch_initial_id.png.
Usage (repo root): python3 experiments/plot_famsw_switch_id.py -> images/ch6/6.5/ch6_famswitch_switch_id.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
SW = [("gauss2t", "Gaussian → $t_4$", "t", "#C2185B"), ("gauss2clayton", "Gaussian → Clayton", "clayton", "#8E6BB8"),
      ("clayton2gumbel", "Clayton → Gumbel", "gumbel", "#E3B505")]
D = {g: pd.read_csv(core.R + f"famsw_ident_{g}.csv") for g, _, _, _ in SW}
for g, _, tgt, _ in SW:
    d = D[g]
    assert len(d) == 60 and (d.t50 >= 0).all() and (d.t90 >= 0).all() and (d.kind == "switch").all() and (d.target == tgt).all()

with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 2, figsize=(core.W, 2.0), sharey=True)
    for ax, col, title in zip(axs, ["t50", "t90"], ["(a) 50% weight", "(b) 90% weight"]):
        data = [D[g][col].values for g, _, _, _ in SW][::-1]    # Gaussian → t4 at the top
        bp = ax.boxplot(data, orientation="horizontal", whis=1.5, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                        capprops=dict(color="#555555", lw=0.8),
                        flierprops=dict(marker="o", ms=2.6, mfc="none", mec="#555555", mew=0.6))
        for patch, (_, _, _, c) in zip(bp["boxes"], SW[::-1]):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#444444"); patch.set_linewidth(0.7)
        ax.set_yticks(range(1, 4)); ax.set_yticklabels([l for _, l, _, _ in SW][::-1], fontsize=7.5)
        ax.set_xlim(0, 1500); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xticks([250, 750, 1250], minor=True)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", which="both", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, pad=3, loc="left")
    fig.text(0.58, 0.02, "Observations after the switch until the new family first reaches the threshold", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.87, bottom=0.23, wspace=0.12)
    fig.savefig("images/ch6/6.5/ch6_famswitch_switch_id.png"); plt.close(fig)

for g, l, _, _ in SW:
    q = lambda x: f"{np.percentile(x, 50):g} [{np.percentile(x, 25):g}, {np.percentile(x, 75):g}]"
    print(f"{l:22s} t50 {q(D[g].t50):24s} t90 {q(D[g].t90):24s} max t90 {D[g].t90.max()}")
print("wrote images/ch6/6.5/ch6_famswitch_switch_id.png")
