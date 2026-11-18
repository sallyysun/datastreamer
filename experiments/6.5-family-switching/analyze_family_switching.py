"""Analyze family identification, adaptation, and predictive accuracy."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import config as C  # noqa: E402
from report_output import write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402

RESULTS = Path("results")
CONSTANT_FAMILIES = [(f, C.FAMILY_LONG_TEXT[f]) for f in C.FAMILIES_T31]
SWITCHES = [(g, f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}") for g, a, b in C.SWITCHES]
ESTIMATORS = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
ACCURACY_METRICS = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10000, 20261115

RMPL_HEADER = ["# Section 6.5 (RMPL / KLL configuration) — family-augmented vs Gaussian-only MFF\n",
          "Configuration: RMPL arms N_eff ∈ {39, 99, 199, 399, 1999} on shared cumulative KLL (k = 256) pseudo-observations; "
          "family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel × 5 memories; each RMPL ρ_S mapped to the family "
          "parameter by inversion, as in the Section 6.4 scoring-function experiment); Gaussian-only MFF = the 5 Gaussian models; "
          "both log loss, η = 1, δ = 10⁻⁴, mixture density forecast, warm-up as Chapter 6. 60 replications (seeds 42–101), "
          "T = 5000, ρ_S ≈ 0.582 constant, switch at observation 2500, evaluation from observation 500.\n",
          "Verification: on the Gaussian constant path, Gaussian-only MFF reproduces the existing Section 6.3 mixture-density MFF "
          "and Section 6.4 log-loss MFF (RMPL) exactly (ρ_S RMSE and excess log loss, all 60 seeds, max difference 0).\n"]



def read_family_results(kind, regime):
    return pd.read_csv(RESULTS / f"famsw_{kind}_{regime}.csv")


def paired_methods(rows, metric, columns="method"):
    return rows.pivot_table(index="seed", columns=columns, values=metric)


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
    return dd.mean(), 2 * standard_error(dd)


def identification():
    M = ["## Initial family identification (constant families, uniform start)\n",
         "| Generating family | 50% weight, median [IQR] | 90% weight, median [IQR] | reached 50% / 90% | another family ≥ 50% after identification | share of later obs. with true family ≥ 50% (median, min) | mean true-family weight, obs. 500–4999 |",
         "|---|---|---|---|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == g) & (p.t >= 500)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {int(d.incorrect_majority.sum())}/60 | "
                 f"{d.share_ge50_after.median():.3f} ({d.share_ge50_after.min():.3f}) | {w:.3f} |")
    M += ["\n## Adaptation following a family switch\n",
          "| Switch | new family 50%, median [IQR] (obs. after switch) | 90%, median [IQR] | reached 50% / 90% | new-family weight before switch | new-family weight, obs. 3000–4999 | another family ≥ 50% after identification |",
          "|---|---|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == d.target.iloc[0]) & (p.t >= 3000)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {d.pre_weight_target.mean():.4f} | {w:.3f} | {int(d.incorrect_majority.sum())}/60 |")
    return M


def density(rows):
    M = ["\n## Predictive density accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE over 60 paired replications\n",
         "| Scenario | first 100 after switch (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | pre-switch (500–2499) | L Gaussian-only / family (post) |",
         "|---|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        x = paired_methods(read_family_results("seeds", g), "excess_x1000", ["method", "window"])
        cells = []
        for w in ["rec_0_100", "late", "post", "pre"]:
            m, se2 = paired(x[("gaussian_mff", w)] - x[("family_mff", w)])
            rows.append(dict(regime=g, window=w, dL=m, dL_2se=se2, n=len(x)))
            cells.append(f"{m:+.1f} ± {se2:.1f}")
        M.append(f"| {gl} | " + " | ".join(cells) + f" | {x[('gaussian_mff', 'post')].mean():.2f} / {x[('family_mff', 'post')].mean():.2f} |")
    M += ["\nConstant families (observations 500–4999): ΔL (negative = cost of family augmentation)\n",
          "| Generating family | ΔL, mean ± 2 SE | L Gaussian-only / family |", "|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "excess_x1000")
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        rows.append(dict(regime=g + "_constant", window="whole", dL=m, dL_2se=se2, n=len(x)))
        M.append(f"| {gl} | {m:+.2f} ± {se2:.2f} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} |")
    return M


def rmse():
    M = ["\nSpearman ρ_S RMSE (×10⁻², whole run): Gaussian-only / family, and difference (positive favours family)\n",
         "| Scenario | RMSE | difference |", "|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES + SWITCHES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "rmse") * 100
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        M.append(f"| {gl} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} | {m:+.2f} ± {se2:.2f} |")
    guard = pd.concat([read_family_results("ident", g) for g, _ in CONSTANT_FAMILIES] + [read_family_results("ident", g) for g, _ in SWITCHES]).n_guard.sum()
    return M + [f"\nNon-finite model log densities replaced by Mff's guard: {int(guard)} in total."]


def accuracy():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for e, _ in ESTIMATORS:
        for g, _ in SWITCHES:
            s = read_family_results(f"seeds_h2_{e}", g)
            for w in ["post", "rec_0_100", "late"]:
                for m, _ in ACCURACY_METRICS:
                    x = paired_methods(s[s.window == w], m)
                    G, F = x.gaussian_mff.values, x.family_mff.values
                    dd = G - F
                    idx = rng.integers(0, len(G), size=(BOOTSTRAP_DRAWS, len(G)))
                    bp = 100 * (G[idx].mean(1) - F[idx].mean(1)) / G[idx].mean(1)
                    rows.append(dict(estimator=e, regime=g, window=w, metric=m, L_gauss=G.mean(), L_family=F.mean(),
                                     dL=dd.mean(), dL_2se=2 * standard_error(dd),
                                     pct=100 * (G.mean() - F.mean()) / G.mean(), pct_2se=2 * bp.std(ddof=1),
                                     pct_lo=np.percentile(bp, 2.5), pct_hi=np.percentile(bp, 97.5), min_gauss=G.min(), n=len(G)))
    return pd.DataFrame(rows)


def row(D, e, g, w, m):
    return D[(D.estimator == e) & (D.regime == g) & (D.window == w) & (D.metric == m)].iloc[0]


def accuracy_report(D):
    M = ["# Section 6.5 predictive accuracy across estimators (family-augmented vs Gaussian-only MFF)\n",
         "Full post-switch window (observations 2500–4999), 60 paired replications. Absolute values ×10⁻³ (nats/obs for excess log loss). "
         "Percentage reduction = 100 (L_G − L_F)/L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means (95% percentile interval in brackets).\n",
         "| Metric | Switch | Estimator | Gaussian-only | Family-augmented | Absolute reduction ± 2 SE | % reduction ± 2 SE [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for m, ml in ACCURACY_METRICS:
        for g, gl in SWITCHES:
            for e, el in ESTIMATORS:
                r = row(D, e, g, "post", m)
                M.append(f"| {ml} | {gl} | {el} | {r.L_gauss:.2f} | {r.L_family:.2f} | {r.dL:+.2f} ± {r.dL_2se:.2f} | "
                         f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}] |")
    post = D[D.window == "post"]
    M.append(f"\nDenominators: smallest per-seed Gaussian-only value across all cells = "
             f"{post[post.metric == 'excess_x1000'].min_gauss.min():.2f} ×10⁻³ (excess) / "
             f"{post[post.metric == 'h2_x1000'].min_gauss.min():.2f} ×10⁻³ (H²); all positive and far from zero.")
    M += ["\n## RMPL: early vs later recovery (absolute reduction ± 2 SE, ×10⁻³; positive favours family augmentation)\n",
          "| Switch | Metric | first 100 (2500–2599) | remaining 2400 (2600–4999) | full post-switch |", "|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        for m, ml in ACCURACY_METRICS:
            c = [row(D, "rmpl", g, w, m) for w in ("rec_0_100", "late", "post")]
            M.append(f"| {gl} | {ml} | " + " | ".join(f"{r.dL:+.2f} ± {r.dL_2se:.2f}" for r in c) + " |")
    M.append("\nH² uses checkpoints: every 10 observations in 2500–2599 (rec_0_100) and every 50 elsewhere; excess log loss uses every observation.")
    M += ["\n## Estimator agreement (full post-switch % reduction: max − min across the three estimators)\n",
          "| Metric | Switch | spread (pp) | largest 2 SE (pp) |", "|---|---|---|---|"]
    for m, ml in ACCURACY_METRICS:
        for g, gl in SWITCHES:
            x = post[(post.regime == g) & (post.metric == m)]
            M.append(f"| {ml} | {gl} | {x.pct.max() - x.pct.min():.2f} | {x.pct_2se.max():.2f} |")
    return M


def write_rmpl():
    rows = []
    lines = RMPL_HEADER + identification() + density(rows) + rmse()
    pd.DataFrame(rows).to_csv(RESULTS / "famsw_rmpl_dl.csv", index=False)
    write_report(RESULTS / "famsw_rmpl_summary_2026-11-15.md", lines)


def write_accuracy():
    results = accuracy()
    results.to_csv(RESULTS / "famsw_accuracy.csv", index=False)
    write_report(RESULTS / "famsw_accuracy_2026-11-15.md", accuracy_report(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("rmpl", "accuracy", "all"), default="all")
    mode = parser.parse_args().mode
    if mode in ("rmpl", "all"):
        write_rmpl()
    if mode in ("accuracy", "all"):
        write_accuracy()


if __name__ == "__main__":
    main()
