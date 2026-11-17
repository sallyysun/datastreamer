"""Section 6.5 on the Chapter 6 configuration (RMPL arms N_eff 39-1999, shared cumulative KLL k = 256 margins, log-loss MFF
with eta = 1, delta = 1e-4, mixture density forecast; 60 seeds 42-101; matched d = 0.6, rho_S ~ 0.582; T = 5000, switch at
2500, evaluation from tick 500). Family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel x 5 memories), each RMPL
rho_S mapped to the family parameter by inversion; Gaussian-only MFF = the 5 Gaussian models. Inputs from probe_strategies
FAMSWITCH=1 (results/famsw_{seeds,ident,path}_<gen>.csv). Delta L = L(Gaussian-only) - L(family), excess log loss x1e-3
nats/obs; positive favours family augmentation. t = observations processed (weights held before observation t).
Usage: python3 famsw_rmpl.py -> results/famsw_rmpl_summary_2026-11-15.md, results/famsw_rmpl_dl.csv,
images/ch6_famswitch_rmpl_weights.png"""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
R = core.R
OUT = "images/"
os.makedirs(OUT, exist_ok=True)
CONST = [("gaussian", "Gaussian"), ("t", "Student's t4"), ("clayton", "Clayton"), ("gumbel", "Gumbel")]
SW = [("gauss2t", r"Gaussian $\rightarrow$ $t_4$", "Gaussian → t4"), ("gauss2clayton", r"Gaussian $\rightarrow$ Clayton", "Gaussian → Clayton"),
      ("clayton2gumbel", r"Clayton $\rightarrow$ Gumbel", "Clayton → Gumbel")]
FAMS = [("gaussian", "Gaussian", "#9a9a9a"), ("t", "$t_4$", "#C2185B"), ("clayton", "Clayton", "#8E6BB8"), ("gumbel", "Gumbel", "#E3B505")]


def rd(kind, g): return pd.read_csv(R + f"famsw_{kind}_{g}.csv")


def q(x):
    x = np.asarray(x, float)
    v = [np.quantile(x, p) for p in (.5, .25, .75)]
    s = lambda a: "never" if not np.isfinite(a) else f"{a:g}"
    return f"{s(v[0])} [{s(v[1])}, {s(v[2])}]"


M = ["# Section 6.5 (RMPL / KLL configuration) — family-augmented vs Gaussian-only MFF\n",
     "Configuration: RMPL arms N_eff ∈ {39, 99, 199, 399, 1999} on shared cumulative KLL (k = 256) pseudo-observations; "
     "family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel × 5 memories; each RMPL ρ_S mapped to the family "
     "parameter by inversion, as in the Section 6.4 scoring-function experiment); Gaussian-only MFF = the 5 Gaussian models; "
     "both log loss, η = 1, δ = 10⁻⁴, mixture density forecast, warm-up as Chapter 6. 60 replications (seeds 42–101), "
     "T = 5000, ρ_S ≈ 0.582 constant, switch at observation 2500, evaluation from observation 500.\n",
     "Verification: on the Gaussian constant path, Gaussian-only MFF reproduces the existing Section 6.3 mixture-density MFF "
     "and Section 6.4 log-loss MFF (RMPL) exactly (ρ_S RMSE and excess log loss, all 60 seeds, max difference 0).\n",
     "## Initial family identification (constant families, uniform start)\n",
     "| Generating family | 50% weight, median [IQR] | 90% weight, median [IQR] | reached 50% / 90% | another family ≥ 50% after identification | share of later obs. with true family ≥ 50% (median, min) | mean true-family weight, obs. 500–4999 |",
     "|---|---|---|---|---|---|---|"]
for g, gl in CONST:
    d = rd("ident", g)
    t50 = np.where(d.t50 >= 0, d.t50, np.inf); t90 = np.where(d.t90 >= 0, d.t90, np.inf)
    p = rd("path", g); w = p[(p.family == g) & (p.t >= 500)]["mean"].mean()
    M.append(f"| {gl} | {q(t50)} | {q(t90)} | {(d.t50 >= 0).sum()}/60 / {(d.t90 >= 0).sum()}/60 | {int(d.incorrect_majority.sum())}/60 | "
             f"{d.share_ge50_after.median():.3f} ({d.share_ge50_after.min():.3f}) | {w:.3f} |")

M += ["\n## Adaptation following a family switch\n",
      "| Switch | new family 50%, median [IQR] (obs. after switch) | 90%, median [IQR] | reached 50% / 90% | new-family weight before switch | new-family weight, obs. 3000–4999 | another family ≥ 50% after identification |",
      "|---|---|---|---|---|---|---|"]
for g, _, gl in SW:
    d = rd("ident", g)
    t50 = np.where(d.t50 >= 0, d.t50, np.inf); t90 = np.where(d.t90 >= 0, d.t90, np.inf)
    p = rd("path", g); tgt = d.target.iloc[0]
    w = p[(p.family == tgt) & (p.t >= 3000)]["mean"].mean()
    M.append(f"| {gl} | {q(t50)} | {q(t90)} | {(d.t50 >= 0).sum()}/60 / {(d.t90 >= 0).sum()}/60 | {d.pre_weight_target.mean():.4f} | {w:.3f} | {int(d.incorrect_majority.sum())}/60 |")

M += ["\n## Predictive density accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE over 60 paired replications\n",
      "| Scenario | first 100 after switch (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | pre-switch (500–2499) | L Gaussian-only / family (post) |",
      "|---|---|---|---|---|---|"]
rows = []
for g, _, gl in SW:
    s = rd("seeds", g)
    x = s.pivot_table(index="seed", columns=["method", "window"], values="excess_x1000")
    cells = []
    for w in ["rec_0_100", "late", "post", "pre"]:
        dd = x[("gaussian_mff", w)] - x[("family_mff", w)]
        m, se2 = dd.mean(), 2 * dd.std(ddof=1) / math.sqrt(len(dd))
        rows.append(dict(regime=g, window=w, dL=m, dL_2se=se2, n=len(dd)))
        cells.append(f"{m:+.1f} ± {se2:.1f}")
    M.append(f"| {gl} | " + " | ".join(cells) + f" | {x[('gaussian_mff', 'post')].mean():.2f} / {x[('family_mff', 'post')].mean():.2f} |")
cost = []
for g, gl in CONST:
    s = rd("seeds", g)
    x = s[s.window == "whole"].pivot_table(index="seed", columns="method", values="excess_x1000")
    dd = x.gaussian_mff - x.family_mff
    m, se2 = dd.mean(), 2 * dd.std(ddof=1) / math.sqrt(len(dd))
    rows.append(dict(regime=g + "_constant", window="whole", dL=m, dL_2se=se2, n=len(dd)))
    cost.append(f"| {gl} | {m:+.2f} ± {se2:.2f} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} |")
M += ["\nConstant families (observations 500–4999): ΔL (negative = cost of family augmentation)\n",
      "| Generating family | ΔL, mean ± 2 SE | L Gaussian-only / family |", "|---|---|---|"] + cost
rmse = []
for g, gl in [(g, gl) for g, gl in CONST] + [(g, gl) for g, _, gl in SW]:
    s = rd("seeds", g); x = s[s.window == "whole"].pivot_table(index="seed", columns="method", values="rmse") * 100
    dd = x.gaussian_mff - x.family_mff
    rmse.append(f"| {gl} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} | {dd.mean():+.2f} ± {2 * dd.std(ddof=1) / math.sqrt(len(dd)):.2f} |")
M += ["\nSpearman ρ_S RMSE (×10⁻², whole run): Gaussian-only / family, and difference (positive favours family)\n",
      "| Scenario | RMSE | difference |", "|---|---|---|"] + rmse
g_all = pd.concat([rd("ident", g) for g, _ in CONST] + [rd("ident", g) for g, _, _ in SW])
M.append(f"\nNon-finite model log densities replaced by Mff's guard: {int(g_all.n_guard.sum())} in total.")
pd.DataFrame(rows).to_csv(R + "famsw_rmpl_dl.csv", index=False)
open(R + "famsw_rmpl_summary_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))

# Figure 1: stacked family weights
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.45), sharey=True)
    for ax, (g, rl, _) in zip(axs, SW):
        p = rd("path", g)
        W = p.pivot_table(index="t", columns="family", values="mean")
        W = W[W.index >= 500][[f for f, _, _ in FAMS]]
        W = W.div(W.sum(axis=1), axis=0)
        ax.stackplot(W.index, *[W[f] for f, _, _ in FAMS], colors=[c for _, _, c in FAMS], lw=0, zorder=2)
        ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
        ax.set_xlim(500, 5000); ax.set_xticks([1000, 2500, 4000]); ax.set_ylim(0, 1)
        ax.set_title(rl, fontsize=8, pad=3)
    axs[0].set_ylabel("Mean MFF weight", fontsize=7.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in FAMS]
    fig.legend(h, [l for _, l, _ in FAMS], loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0),
               handlelength=1.2, columnspacing=1.4)
    fig.text(0.54, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.18, wspace=0.08)
    fig.savefig(OUT + "ch6_famswitch_rmpl_weights.png"); plt.close(fig)

print("wrote", OUT + "ch6_famswitch_rmpl_weights.png")
