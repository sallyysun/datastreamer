"""Section 6.5, all estimators: post-switch (ticks 2500-4999) reduction in excess log loss, Delta L = L(Gaussian-only MFF) -
L(family-augmented MFF), x1e-3 nats/obs, mean ± 2 SE over 60 paired replications, for RMPL, Counting Matrix and Hermite
Series (each estimator's five arms N_eff 39-1999, rho_S mapped to Gaussian / t4 / Clayton / Gumbel by inversion; log loss,
eta = 1, delta = 1e-4; mixture forecast). Inputs: probe_strategies FAMSWITCH=1 [FS_EST=xiao|hermite]:
results/famsw_seeds_<switch>.csv (RMPL), results/famsw_seeds_<est>_<switch>.csv.
Usage (repo root): python3 experiments/famsw_allest.py -> images/ch6/6.5/ch6_famswitch_density_allest.png,
results/famsw_allest_dl.csv"""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
R = core.R
SW = [("gauss2t", "Gaussian →\n$t_4$"), ("gauss2clayton", "Gaussian →\nClayton"), ("clayton2gumbel", "Clayton →\nGumbel")]
EST = [("rmpl", "RMPL", "mle", ""), ("xiao", "Counting Matrix", "xiao", "xiao_"), ("hermite", "Hermite Series", "hermite", "hermite_")]

rows = []
for e, el, _, pre in EST:
    for g, _ in SW:
        s = pd.read_csv(R + f"famsw_seeds_{pre}{g}.csv")
        x = s.pivot_table(index="seed", columns=["method", "window"], values="excess_x1000")
        for w in ["rec_0_100", "late", "post"]:
            dd = x[("gaussian_mff", w)] - x[("family_mff", w)]
            rows.append(dict(estimator=e, regime=g, window=w, dL=dd.mean(), dL_2se=2 * dd.std(ddof=1) / math.sqrt(len(dd)), n=len(dd),
                             L_gauss=x[("gaussian_mff", w)].mean(), L_family=x[("family_mff", w)].mean()))
D = pd.DataFrame(rows)
D.to_csv(R + "famsw_allest_dl.csv", index=False)
print(D.round(2).to_string(index=False))

with plt.rc_context(ce.RC):
    fig, ax = plt.subplots(figsize=(core.W * 0.72, 2.45))
    bw = 0.26
    for j, (e, el, ck, _) in enumerate(EST):
        for i, (g, _) in enumerate(SW):
            r = D[(D.estimator == e) & (D.regime == g) & (D.window == "post")].iloc[0]
            x = i + (j - 1) * bw
            ax.bar(x, r.dL, width=bw * 0.92, color=ce.COL[ck], zorder=3)
            ax.errorbar(x, r.dL, yerr=r.dL_2se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
    ax.axhline(0, color="#222222", lw=0.7, zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in SW], fontsize=7); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.55)
    post = D[D.window == "post"]
    ax.set_ylim(0, (post.dL + post.dL_2se).max() * 1.1)
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel("Reduction in excess log loss\n($\\times10^{-3}$ nats per observation)", fontsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, color=ce.COL[ck]) for _, _, ck, _ in EST]
    fig.legend(h, [el for _, el, _, _ in EST], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.57, 1.0),
               handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.19)
    fig.savefig("images/ch6/6.5/ch6_famswitch_density_allest.png"); plt.close(fig)
print("wrote images/ch6/6.5/ch6_famswitch_density_allest.png")
