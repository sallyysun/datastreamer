"""Section 6.5 predictive accuracy across estimators: family-augmented vs Gaussian-only MFF (RMPL / Counting Matrix / Hermite
Series arms N_eff 39-1999 on the Chapter 6 configuration; 20 vs 5 models; log loss, eta = 1, delta = 1e-4; mixture forecast),
after the three switches, full post-switch window (ticks 2500-4999). Metrics: excess log loss at the true PIT (per-tick mean)
and squared Hellinger H^2 to the true copula on the 6.3 grid (checkpoints every 50; every 10 in 2500-2599 counted in
rec_0_100 only). Percentage reduction = 100 (mean_G - mean_F) / mean_G; ± 2 SE from a paired-seed bootstrap of the ratio of
means (B = 10000, seed 20261115), with the 95% percentile interval. Absolute reduction mean ± 2 SE (paired).
Inputs: probe_strategies FAMSWITCH=1 FS_H2=1 [FS_EST=...] -> results/famsw_seeds_h2_<est>_<switch>.csv.
Usage: python3 famsw_accuracy.py -> images/ch6_famswitch_accuracy.png,
results/famsw_accuracy.csv, results/famsw_accuracy_2026-11-15.md"""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
R = core.R
SW = [(g, f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}", f"{C.FAMILY_MATH[a]} →\n{C.FAMILY_MATH[b]}") for g, a, b in C.SWITCHES]
EST = [(e, C.EST_LABEL[e], C.EST_COLOUR_KEY[e]) for e in C.ESTIMATORS]
MET = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]
rng = np.random.default_rng(20261115)
B = 10000
rows = []


def load(e, g):
    s = pd.read_csv(R + f"famsw_seeds_h2_{e}_{g}.csv")
    return s


for e, el, _ in EST:
    for g, gl, _ in SW:
        s = load(e, g)
        for w in ["post", "rec_0_100", "late"]:
            for m, ml in MET:
                x = s[s.window == w].pivot_table(index="seed", columns="method", values=m)
                G, F = x.gaussian_mff.values, x.family_mff.values
                dd = G - F
                pct = 100 * (G.mean() - F.mean()) / G.mean()
                idx = rng.integers(0, len(G), size=(B, len(G)))
                bp = 100 * (G[idx].mean(1) - F[idx].mean(1)) / G[idx].mean(1)
                rows.append(dict(estimator=e, regime=g, window=w, metric=m, L_gauss=G.mean(), L_family=F.mean(),
                                 dL=dd.mean(), dL_2se=2 * dd.std(ddof=1) / math.sqrt(len(dd)), pct=pct, pct_2se=2 * bp.std(ddof=1),
                                 pct_lo=np.percentile(bp, 2.5), pct_hi=np.percentile(bp, 97.5), min_gauss=G.min(), n=len(G)))
D = pd.DataFrame(rows)
D.to_csv(R + "famsw_accuracy.csv", index=False)

M = ["# Section 6.5 predictive accuracy across estimators (family-augmented vs Gaussian-only MFF)\n",
     "Full post-switch window (observations 2500–4999), 60 paired replications. Absolute values ×10⁻³ (nats/obs for excess log loss). "
     "Percentage reduction = 100 (L_G − L_F)/L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means (95% percentile interval in brackets).\n",
     "| Metric | Switch | Estimator | Gaussian-only | Family-augmented | Absolute reduction ± 2 SE | % reduction ± 2 SE [95% CI] |",
     "|---|---|---|---|---|---|---|"]
for m, ml in MET:
    for g, gl, _ in SW:
        for e, el, _ in EST:
            r = D[(D.estimator == e) & (D.regime == g) & (D.window == "post") & (D.metric == m)].iloc[0]
            M.append(f"| {ml} | {gl} | {el} | {r.L_gauss:.2f} | {r.L_family:.2f} | {r.dL:+.2f} ± {r.dL_2se:.2f} | "
                     f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}] |")
M.append(f"\nDenominators: smallest per-seed Gaussian-only value across all cells = "
         f"{D[(D.window == 'post') & (D.metric == 'excess_x1000')].min_gauss.min():.2f} ×10⁻³ (excess) / "
         f"{D[(D.window == 'post') & (D.metric == 'h2_x1000')].min_gauss.min():.2f} ×10⁻³ (H²); all positive and far from zero.")
M += ["\n## RMPL: early vs later recovery (absolute reduction ± 2 SE, ×10⁻³; positive favours family augmentation)\n",
      "| Switch | Metric | first 100 (2500–2599) | remaining 2400 (2600–4999) | full post-switch |", "|---|---|---|---|---|"]
for g, gl, _ in SW:
    for m, ml in MET:
        c = [D[(D.estimator == "rmpl") & (D.regime == g) & (D.window == w) & (D.metric == m)].iloc[0] for w in ("rec_0_100", "late", "post")]
        M.append(f"| {gl} | {ml} | " + " | ".join(f"{r.dL:+.2f} ± {r.dL_2se:.2f}" for r in c) + " |")
M.append("\nH² uses checkpoints: every 10 observations in 2500–2599 (rec_0_100) and every 50 elsewhere; excess log loss uses every observation.")
M += ["\n## Estimator agreement (full post-switch % reduction: max − min across the three estimators)\n", "| Metric | Switch | spread (pp) | largest 2 SE (pp) |", "|---|---|---|---|"]
for m, ml in MET:
    for g, gl, _ in SW:
        x = D[(D.regime == g) & (D.window == "post") & (D.metric == m)]
        M.append(f"| {ml} | {gl} | {x.pct.max() - x.pct.min():.2f} | {x.pct_2se.max():.2f} |")
open(R + "famsw_accuracy_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))

with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 2, figsize=(core.W, 2.5), sharey=False)
    bw = 0.26
    for ax, (m, ml), tag in zip(axs, MET, ["(a)", "(b)"]):
        top = 0
        for j, (e, el, ck) in enumerate(EST):
            for i, (g, _, gx) in enumerate(SW):
                r = D[(D.estimator == e) & (D.regime == g) & (D.window == "post") & (D.metric == m)].iloc[0]
                x = i + (j - 1) * bw
                ax.bar(x, r.pct, width=bw * 0.92, color=ce.COL[ck], zorder=3)
                ax.errorbar(x, r.pct, yerr=r.pct_2se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
                top = max(top, r.pct + r.pct_2se)
        ax.axhline(0, color="#222222", lw=0.7, zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels([gx for _, _, gx in SW], fontsize=7); ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(f"{tag} {ml}", fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("Reduction relative to\nGaussian-only MFF (%)", fontsize=7.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=ce.COL[ck]) for _, _, ck in EST]
    fig.legend(h, [el for _, el, _ in EST], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
               handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.8, bottom=0.18, wspace=0.18)
    fig.savefig("images/ch6_famswitch_accuracy.png"); plt.close(fig)
print("wrote images/ch6_famswitch_accuracy.png")
