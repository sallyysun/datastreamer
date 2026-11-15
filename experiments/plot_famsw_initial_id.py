"""Section 6.5 initial family identification, RMPL / KLL configuration: box plots of the observations needed for the true
family's weight (summed over the five memories N_eff 39-1999) to first reach 50% (a) and 90% (b) of the family-augmented
MFF weight, on the four constant-family paths, 60 replications each. Data: results/famsw_ident_<family>.csv
(probe_strategies FAMSWITCH=1, kind = initial; first-crossing times, all 60 reach both thresholds). Conventional box plots:
median, IQR, whiskers at 1.5 IQR, individual outliers; quartiles by linear interpolation (numpy default), as in the table.
Usage (repo root): python3 experiments/plot_famsw_initial_id.py -> images/ch6/6.5/ch6_famswitch_initial_id.png"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
FAMS = [("gaussian", "Gaussian", "#9a9a9a"), ("t", "Student's $t_4$", "#C2185B"), ("clayton", "Clayton", "#8E6BB8"), ("gumbel", "Gumbel", "#E3B505")]
D = {f: pd.read_csv(core.R + f"famsw_ident_{f}.csv") for f, _, _ in FAMS}
for f, _, _ in FAMS:
    assert len(D[f]) == 60 and (D[f].t50 >= 0).all() and (D[f].t90 >= 0).all() and (D[f].kind == "initial").all()

with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 2, figsize=(core.W, 2.3), sharey=True)
    for ax, col, title in zip(axs, ["t50", "t90"], ["(a) 50% weight", "(b) 90% weight"]):
        data = [D[f][col].values for f, _, _ in FAMS][::-1]    # Gaussian at the top
        bp = ax.boxplot(data, orientation="horizontal", whis=1.5, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color="#111111", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8),
                        capprops=dict(color="#555555", lw=0.8),
                        flierprops=dict(marker="o", ms=2.6, mfc="none", mec="#555555", mew=0.6))
        for patch, (_, _, c) in zip(bp["boxes"], FAMS[::-1]):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#444444"); patch.set_linewidth(0.7)
        ax.set_yticks(range(1, 5)); ax.set_yticklabels([l for _, l, _ in FAMS][::-1], fontsize=7.5)
        ax.set_xlim(0, 1500); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xticks([250, 750, 1250], minor=True)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", which="both", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, pad=3, loc="left")
    fig.text(0.56, 0.02, "Observations until the true family first reaches the threshold", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.2, wspace=0.12)
    fig.savefig("images/ch6/6.5/ch6_famswitch_initial_id.png"); plt.close(fig)

for f, l, _ in FAMS:
    q = lambda x: f"{np.percentile(x, 50):g} [{np.percentile(x, 25):g}, {np.percentile(x, 75):g}]"
    print(f"{l:16s} t50 {q(D[f].t50):22s} t90 {q(D[f].t90)}")
print("wrote images/ch6/6.5/ch6_famswitch_initial_id.png")
