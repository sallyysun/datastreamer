"""Section 6.5 on the RMPL configuration: family-augmented MFF (Gaussian, t4, Clayton, Gumbel x 5 memories, each arm's
rho_S inverted into the family parameter) vs Gaussian-only MFF, from results/famsw_{seeds,ident,path}_<gen>.csv
(scenarios FAMSWITCH=1): initial identification, adaptation after a switch, and Delta L = L(Gaussian-only) -
L(family), positive favouring augmentation. Writes results/famsw_rmpl_summary_2026-11-15.md, results/famsw_rmpl_dl.csv
and images/ch6_famswitch_rmpl_weights.png."""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
R = ce.R
OUT = "images/"
CONST = [(f, C.FAMILY_LONG_TEXT[f]) for f in C.FAMILIES_T31]
SW = [(g, f"{C.FAMILY_MATH[a]} $\\rightarrow$ {C.FAMILY_MATH[b]}", f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}") for g, a, b in C.SWITCHES]
FAMS = [(f, C.FAMILY_MATH[f], C.FAMILY_COLOUR[f]) for f in C.FAMILIES_T31]
HEADER = ["# Section 6.5 (RMPL / KLL configuration) — family-augmented vs Gaussian-only MFF\n",
          "Configuration: RMPL arms N_eff ∈ {39, 99, 199, 399, 1999} on shared cumulative KLL (k = 256) pseudo-observations; "
          "family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel × 5 memories; each RMPL ρ_S mapped to the family "
          "parameter by inversion, as in the Section 6.4 scoring-function experiment); Gaussian-only MFF = the 5 Gaussian models; "
          "both log loss, η = 1, δ = 10⁻⁴, mixture density forecast, warm-up as Chapter 6. 60 replications (seeds 42–101), "
          "T = 5000, ρ_S ≈ 0.582 constant, switch at observation 2500, evaluation from observation 500.\n",
          "Verification: on the Gaussian constant path, Gaussian-only MFF reproduces the existing Section 6.3 mixture-density MFF "
          "and Section 6.4 log-loss MFF (RMPL) exactly (ρ_S RMSE and excess log loss, all 60 seeds, max difference 0).\n"]


def rd(kind, g):
    return pd.read_csv(R + f"famsw_{kind}_{g}.csv")


def q(x):
    def s(a):
        return "never" if not np.isfinite(a) else f"{a:g}"
    v = [np.quantile(np.asarray(x, float), p) for p in (.5, .25, .75)]
    return f"{s(v[0])} [{s(v[1])}, {s(v[2])}]"


def crossing(d):
    t50 = np.where(d.t50 >= 0, d.t50, np.inf)
    t90 = np.where(d.t90 >= 0, d.t90, np.inf)
    return f"{q(t50)} | {q(t90)} | {(d.t50 >= 0).sum()}/60 / {(d.t90 >= 0).sum()}/60"


def paired(dd):
    return dd.mean(), 2 * dd.std(ddof=1) / math.sqrt(len(dd))


def identification():
    M = ["## Initial family identification (constant families, uniform start)\n",
         "| Generating family | 50% weight, median [IQR] | 90% weight, median [IQR] | reached 50% / 90% | another family ≥ 50% after identification | share of later obs. with true family ≥ 50% (median, min) | mean true-family weight, obs. 500–4999 |",
         "|---|---|---|---|---|---|---|"]
    for g, gl in CONST:
        d, p = rd("ident", g), rd("path", g)
        w = p[(p.family == g) & (p.t >= 500)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {int(d.incorrect_majority.sum())}/60 | "
                 f"{d.share_ge50_after.median():.3f} ({d.share_ge50_after.min():.3f}) | {w:.3f} |")
    M += ["\n## Adaptation following a family switch\n",
          "| Switch | new family 50%, median [IQR] (obs. after switch) | 90%, median [IQR] | reached 50% / 90% | new-family weight before switch | new-family weight, obs. 3000–4999 | another family ≥ 50% after identification |",
          "|---|---|---|---|---|---|---|"]
    for g, _, gl in SW:
        d, p = rd("ident", g), rd("path", g)
        w = p[(p.family == d.target.iloc[0]) & (p.t >= 3000)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {d.pre_weight_target.mean():.4f} | {w:.3f} | {int(d.incorrect_majority.sum())}/60 |")
    return M


def density(rows):
    M = ["\n## Predictive density accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE over 60 paired replications\n",
         "| Scenario | first 100 after switch (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | pre-switch (500–2499) | L Gaussian-only / family (post) |",
         "|---|---|---|---|---|---|"]
    for g, _, gl in SW:
        x = rd("seeds", g).pivot_table(index="seed", columns=["method", "window"], values="excess_x1000")
        cells = []
        for w in ["rec_0_100", "late", "post", "pre"]:
            m, se2 = paired(x[("gaussian_mff", w)] - x[("family_mff", w)])
            rows.append(dict(regime=g, window=w, dL=m, dL_2se=se2, n=len(x)))
            cells.append(f"{m:+.1f} ± {se2:.1f}")
        M.append(f"| {gl} | " + " | ".join(cells) + f" | {x[('gaussian_mff', 'post')].mean():.2f} / {x[('family_mff', 'post')].mean():.2f} |")
    M += ["\nConstant families (observations 500–4999): ΔL (negative = cost of family augmentation)\n",
          "| Generating family | ΔL, mean ± 2 SE | L Gaussian-only / family |", "|---|---|---|"]
    for g, gl in CONST:
        s = rd("seeds", g)
        x = s[s.window == "whole"].pivot_table(index="seed", columns="method", values="excess_x1000")
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        rows.append(dict(regime=g + "_constant", window="whole", dL=m, dL_2se=se2, n=len(x)))
        M.append(f"| {gl} | {m:+.2f} ± {se2:.2f} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} |")
    return M


def rmse():
    M = ["\nSpearman ρ_S RMSE (×10⁻², whole run): Gaussian-only / family, and difference (positive favours family)\n",
         "| Scenario | RMSE | difference |", "|---|---|---|"]
    for g, gl in CONST + [(g, gl) for g, _, gl in SW]:
        s = rd("seeds", g)
        x = s[s.window == "whole"].pivot_table(index="seed", columns="method", values="rmse") * 100
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        M.append(f"| {gl} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} | {m:+.2f} ± {se2:.2f} |")
    guard = pd.concat([rd("ident", g) for g, _ in CONST] + [rd("ident", g) for g, _, _ in SW]).n_guard.sum()
    return M + [f"\nNon-finite model log densities replaced by Mff's guard: {int(guard)} in total."]


def weights_figure():
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.45), sharey=True)
        for ax, (g, rl, _) in zip(axs, SW):
            W = rd("path", g).pivot_table(index="t", columns="family", values="mean")
            W = W[W.index >= 500][[f for f, _, _ in FAMS]]
            W = W.div(W.sum(axis=1), axis=0)
            ax.stackplot(W.index, *[W[f] for f, _, _ in FAMS], colors=[c for _, _, c in FAMS], lw=0, zorder=2)
            ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
            ax.set_xlim(500, 5000)
            ax.set_xticks([1000, 2500, 4000])
            ax.set_ylim(0, 1)
            ax.set_title(rl, fontsize=8, pad=3)
        axs[0].set_ylabel("Mean MFF weight", fontsize=7.5)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in FAMS]
        fig.legend(h, [l for _, l, _ in FAMS], loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
                   handlelength=1.2, columnspacing=1.4)
        fig.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.18, wspace=0.08)
        fig.savefig(OUT + "ch6_famswitch_rmpl_weights.png")
        plt.close(fig)
    print("wrote", OUT + "ch6_famswitch_rmpl_weights.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    M = HEADER + identification() + density(rows) + rmse()
    pd.DataFrame(rows).to_csv(R + "famsw_rmpl_dl.csv", index=False)
    open(R + "famsw_rmpl_summary_2026-11-15.md", "w").write("\n".join(M) + "\n")
    print("\n".join(M))
    weights_figure()


if __name__ == "__main__":
    main()
