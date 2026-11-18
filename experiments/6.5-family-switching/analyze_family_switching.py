"""Analyze family identification, adaptation, and predictive accuracy."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import config as C  # noqa: E402
from report_output import format_estimate, reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402

RESULTS = Path("results")
CONSTANT_FAMILIES = [(f, C.FAMILY_LONG_TEXT[f]) for f in C.FAMILIES_T31]
SWITCHES = [(g, f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}") for g, a, b in C.SWITCHES]
ESTIMATORS = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
ACCURACY_METRICS = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10000, 20261115
EARLY_END = C.CHANGE + 100
SETTLED_START = C.CHANGE + 500
EVALUATION_TICKS = f"{C.BURN}–{C.T - 1}"
EARLY_TICKS = f"{C.CHANGE}–{EARLY_END - 1}"
LATE_TICKS = f"{EARLY_END}–{C.T - 1}"
POST_TICKS = f"{C.CHANGE}–{C.T - 1}"
PRE_TICKS = f"{C.BURN}–{C.CHANGE - 1}"
SETTLED_TICKS = f"{SETTLED_START}–{C.T - 1}"

RMPL_HEADER = ["# Section 6.5 (RMPL / KLL configuration) — family-augmented vs Gaussian-only MFF\n",
          f"RMPL memories {', '.join(map(str, C.MEMORIES))} use shared cumulative KLL "
          f"(k = {C.STUDY['kll_capacity']}). Family-augmented MFF combines {len(CONSTANT_FAMILIES) * len(C.MEMORIES)} models "
          f"by mapping each RMPL ρ_S to the working family's parameter. Gaussian-only MFF uses {len(C.MEMORIES)} models. "
          "Both use mixture forecasts, log loss, η = 1 and δ = 10⁻⁴. "
          f"Study defaults are {len(C.SEEDS)} replications from seed {C.SEEDS.start}, T = {C.T}, "
          f"ρ_S = {C.CONSTANT_RHO_S:g}, switch at {C.CHANGE}, and evaluation from {C.BURN}.\n",
          "In the reference study, Gaussian-only MFF on the constant Gaussian path exactly matched the Section 6.3 mixture-density "
          "and Section 6.4 log-loss RMPL results for ρ_S RMSE and excess log loss.\n"]


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
    return f"{q(t50)} | {q(t90)} | {(d.t50 >= 0).sum()}/{len(d)} / {(d.t90 >= 0).sum()}/{len(d)}"


def paired(dd):
    return dd.mean(), 2 * standard_error(dd)


def identification():
    M = ["## Initial identification (constant family, uniform start)\n",
         "| Generating family | 50% weight, median [IQR] | 90% weight, median [IQR] | reached 50% / 90% | another family ≥ 50% after identification | "
         f"share of later obs. with true family ≥ 50% (median, min) | mean true-family weight, obs. {EVALUATION_TICKS} |",
         "|---|---|---|---|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == g) & (p.t >= C.BURN)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {int(d.incorrect_majority.sum())}/{len(d)} | "
                 f"{d.share_ge50_after.median():.3f} ({d.share_ge50_after.min():.3f}) | {w:.3f} |")
    M += ["\n## Switch adaptation\n",
          "| Switch | new family 50%, median [IQR] (obs. after switch) | 90%, median [IQR] | reached 50% / 90% | new-family weight before switch | "
          f"new-family weight, obs. {SETTLED_TICKS} | another family ≥ 50% after identification |",
          "|---|---|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == d.target.iloc[0]) & (p.t >= SETTLED_START)]["mean"].mean()
        M.append(f"| {gl} | {crossing(d)} | {d.pre_weight_target.mean():.4f} | {w:.3f} | {int(d.incorrect_majority.sum())}/{len(d)} |")
    return M


def density(rows):
    M = ["\n## Predictive density accuracy\n",
         "ΔL is Gaussian-only minus family-augmented loss (×10⁻³ nats/obs, paired mean ± 2 SE).\n",
         f"| Scenario | early recovery ({EARLY_TICKS}) | later recovery ({LATE_TICKS}) | full post-switch ({POST_TICKS}) | "
         f"pre-switch ({PRE_TICKS}) | L Gaussian-only / family (post) |",
         "|---|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        x = paired_methods(read_family_results("seeds", g), "excess_x1000", ["method", "window"])
        cells = []
        for w in ["rec_0_100", "late", "post", "pre"]:
            m, se2 = paired(x[("gaussian_mff", w)] - x[("family_mff", w)])
            rows.append(dict(regime=g, window=w, dL=m, dL_2se=se2, n=len(x)))
            cells.append(format_estimate(m, se2, "+.1f", ".1f"))
        M.append(f"| {gl} | " + " | ".join(cells) + f" | {x[('gaussian_mff', 'post')].mean():.2f} / {x[('family_mff', 'post')].mean():.2f} |")
    M += [f"\nConstant families (observations {EVALUATION_TICKS}): ΔL (negative = cost of family augmentation)\n",
          "| Generating family | ΔL, mean ± 2 SE | L Gaussian-only / family |", "|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "excess_x1000")
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        rows.append(dict(regime=g + "_constant", window="whole", dL=m, dL_2se=se2, n=len(x)))
        M.append(f"| {gl} | {format_estimate(m, se2)} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} |")
    return M


def rmse():
    M = ["\nSpearman ρ_S RMSE (×10⁻², whole run): Gaussian-only / family, and difference (positive favours family)\n",
         "| Scenario | RMSE | difference |", "|---|---|---|"]
    for g, gl in CONSTANT_FAMILIES + SWITCHES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "rmse") * 100
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        M.append(f"| {gl} | {x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f} | {format_estimate(m, se2)} |")
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
         f"Full post-switch window (observations {POST_TICKS}). Absolute values ×10⁻³ (nats/obs for excess log loss). "
         "Percentage reduction = 100 (L_G − L_F)/L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means (95% percentile interval in brackets).\n",
         "| Metric | Switch | Estimator | Gaussian-only | Family-augmented | Absolute reduction ± 2 SE | % reduction ± 2 SE [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for m, ml in ACCURACY_METRICS:
        for g, gl in SWITCHES:
            for e, el in ESTIMATORS:
                r = row(D, e, g, "post", m)
                M.append(f"| {ml} | {gl} | {el} | {r.L_gauss:.2f} | {r.L_family:.2f} | {format_estimate(r.dL, r.dL_2se)} | "
                         f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}] |")
    post = D[D.window == "post"]
    M.append(f"\nDenominators: smallest per-seed Gaussian-only value across all cells = "
             f"{post[post.metric == 'excess_x1000'].min_gauss.min():.2f} ×10⁻³ (excess) / "
             f"{post[post.metric == 'h2_x1000'].min_gauss.min():.2f} ×10⁻³ (H²); all positive and far from zero.")
    M += ["\n## RMPL recovery windows\n",
          "Absolute reduction (×10⁻³, mean ± 2 SE). Positive values favour family augmentation.\n",
          f"| Switch | Metric | early ({EARLY_TICKS}) | later ({LATE_TICKS}) | full post-switch |", "|---|---|---|---|---|"]
    for g, gl in SWITCHES:
        for m, ml in ACCURACY_METRICS:
            c = [row(D, "rmpl", g, w, m) for w in ("rec_0_100", "late", "post")]
            M.append(f"| {gl} | {ml} | " + " | ".join(format_estimate(r.dL, r.dL_2se) for r in c) + " |")
    M.append(f"\nH² uses checkpoints every 10 observations in {EARLY_TICKS} and every 50 elsewhere. Excess log loss uses every observation.")
    M += ["\n## Estimator agreement after the switch\n",
          "| Metric | Switch | spread (pp) | largest 2 SE (pp) |", "|---|---|---|---|"]
    for m, ml in ACCURACY_METRICS:
        for g, gl in SWITCHES:
            x = post[(post.regime == g) & (post.metric == m)]
            M.append(f"| {ml} | {gl} | {x.pct.max() - x.pct.min():.2f} | {x.pct_2se.max():.2f} |")
    return M


def write_rmpl():
    rows = []
    density_lines = density(rows)
    pd.DataFrame(rows).to_csv(RESULTS / "famsw_rmpl_dl.csv", index=False)
    if reports_enabled():
        write_report(RESULTS / "famsw_rmpl_summary_2026-11-15.md", RMPL_HEADER + identification() + density_lines + rmse())


def write_accuracy():
    results = accuracy()
    results.to_csv(RESULTS / "famsw_accuracy.csv", index=False)
    if reports_enabled():
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
