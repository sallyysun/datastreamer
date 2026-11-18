"""Section 6.5 predictive accuracy across estimators: family-augmented (20 models) vs Gaussian-only (5 models) MFF after
the three family switches, from results/famsw_seeds_h2_<est>_<switch>.csv (scenarios FAMSWITCH=1 FS_H2=1).
Percentage reduction = 100 (L_G - L_F) / L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means.
Writes results/famsw_accuracy.csv, results/famsw_accuracy_2026-11-15.md and images/ch6_famswitch_accuracy.png."""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
R = ce.R
SW = [(g, f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}", f"{C.FAMILY_MATH[a]} →\n{C.FAMILY_MATH[b]}") for g, a, b in C.SWITCHES]
EST = [(e, C.EST_LABEL[e], C.EST_COLOUR_KEY[e]) for e in C.ESTIMATORS]
MET = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]
BOOT, BOOT_SEED = 10000, 20261115


def accuracy():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for e, _, _ in EST:
        for g, _, _ in SW:
            s = pd.read_csv(R + f"famsw_seeds_h2_{e}_{g}.csv")
            for w in ["post", "rec_0_100", "late"]:
                for m, _ in MET:
                    x = s[s.window == w].pivot_table(index="seed", columns="method", values=m)
                    G, F = x.gaussian_mff.values, x.family_mff.values
                    dd = G - F
                    idx = rng.integers(0, len(G), size=(BOOT, len(G)))
                    bp = 100 * (G[idx].mean(1) - F[idx].mean(1)) / G[idx].mean(1)
                    rows.append(dict(estimator=e, regime=g, window=w, metric=m, L_gauss=G.mean(), L_family=F.mean(),
                                     dL=dd.mean(), dL_2se=2 * dd.std(ddof=1) / math.sqrt(len(dd)),
                                     pct=100 * (G.mean() - F.mean()) / G.mean(), pct_2se=2 * bp.std(ddof=1),
                                     pct_lo=np.percentile(bp, 2.5), pct_hi=np.percentile(bp, 97.5), min_gauss=G.min(), n=len(G)))
    return pd.DataFrame(rows)


def row(D, e, g, w, m):
    return D[(D.estimator == e) & (D.regime == g) & (D.window == w) & (D.metric == m)].iloc[0]


def report(D):
    M = ["# Section 6.5 predictive accuracy across estimators (family-augmented vs Gaussian-only MFF)\n",
         "Full post-switch window (observations 2500–4999), 60 paired replications. Absolute values ×10⁻³ (nats/obs for excess log loss). "
         "Percentage reduction = 100 (L_G − L_F)/L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means (95% percentile interval in brackets).\n",
         "| Metric | Switch | Estimator | Gaussian-only | Family-augmented | Absolute reduction ± 2 SE | % reduction ± 2 SE [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for m, ml in MET:
        for g, gl, _ in SW:
            for e, el, _ in EST:
                r = row(D, e, g, "post", m)
                M.append(f"| {ml} | {gl} | {el} | {r.L_gauss:.2f} | {r.L_family:.2f} | {r.dL:+.2f} ± {r.dL_2se:.2f} | "
                         f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}] |")
    post = D[D.window == "post"]
    M.append(f"\nDenominators: smallest per-seed Gaussian-only value across all cells = "
             f"{post[post.metric == 'excess_x1000'].min_gauss.min():.2f} ×10⁻³ (excess) / "
             f"{post[post.metric == 'h2_x1000'].min_gauss.min():.2f} ×10⁻³ (H²); all positive and far from zero.")
    M += ["\n## RMPL: early vs later recovery (absolute reduction ± 2 SE, ×10⁻³; positive favours family augmentation)\n",
          "| Switch | Metric | first 100 (2500–2599) | remaining 2400 (2600–4999) | full post-switch |", "|---|---|---|---|---|"]
    for g, gl, _ in SW:
        for m, ml in MET:
            c = [row(D, "rmpl", g, w, m) for w in ("rec_0_100", "late", "post")]
            M.append(f"| {gl} | {ml} | " + " | ".join(f"{r.dL:+.2f} ± {r.dL_2se:.2f}" for r in c) + " |")
    M.append("\nH² uses checkpoints: every 10 observations in 2500–2599 (rec_0_100) and every 50 elsewhere; excess log loss uses every observation.")
    M += ["\n## Estimator agreement (full post-switch % reduction: max − min across the three estimators)\n",
          "| Metric | Switch | spread (pp) | largest 2 SE (pp) |", "|---|---|---|---|"]
    for m, ml in MET:
        for g, gl, _ in SW:
            x = post[(post.regime == g) & (post.metric == m)]
            M.append(f"| {ml} | {gl} | {x.pct.max() - x.pct.min():.2f} | {x.pct_2se.max():.2f} |")
    return M


def figure(D):
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 2, figsize=(ce.W, 2.5), sharey=False)
        for ax, (m, ml), tag in zip(axs, MET, ["(a)", "(b)"]):
            for j, (e, _, ck) in enumerate(EST):
                for i, (g, _, _) in enumerate(SW):
                    r = row(D, e, g, "post", m)
                    x = i + (j - 1) * bw
                    ax.bar(x, r.pct, width=bw * 0.92, color=ce.COL[ck], zorder=3)
                    ax.errorbar(x, r.pct, yerr=r.pct_2se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
            ax.axhline(0, color="#222222", lw=0.7, zorder=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels([gx for _, _, gx in SW], fontsize=7)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.55, 2.55)
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(f"{tag} {ml}", fontsize=8, pad=3, loc="left")
        axs[0].set_ylabel("Reduction relative to\nGaussian-only MFF (%)", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color=ce.COL[ck]) for _, _, ck in EST]
        fig.legend(h, [el for _, el, _ in EST], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
                   handlelength=1.2, columnspacing=1.4)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.8, bottom=0.18, wspace=0.18)
        fig.savefig("images/ch6_famswitch_accuracy.png")
        plt.close(fig)
    print("wrote images/ch6_famswitch_accuracy.png")


def main():
    D = accuracy()
    D.to_csv(R + "famsw_accuracy.csv", index=False)
    M = report(D)
    open(R + "famsw_accuracy_2026-11-15.md", "w").write("\n".join(M) + "\n")
    print("\n".join(M))
    figure(D)


if __name__ == "__main__":
    main()
