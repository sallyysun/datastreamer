"""Analyse family identification, adaptation, and predictive accuracy."""
import itertools
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

RMPL_HEADER = ["# Section 6.5 — family-augmented vs Gaussian-only MFF (RMPL, KLL)\n",
               f"Family MFF combines {len(CONSTANT_FAMILIES) * len(C.MEMORIES)} models, Gaussian-only {len(C.MEMORIES)}. "
               f"Mixture forecasts, log loss, η = 1, δ = 10⁻⁴; {len(C.SEEDS)} seeds, switch at {C.CHANGE}, scored from {C.BURN}.\n"]


def read_family_results(kind, regime):
    return pd.read_csv(RESULTS / f"famsw_{kind}_{regime}.csv")


def paired_methods(rows, metric, columns="method"):
    return rows.pivot_table(index="seed", columns=columns, values=metric)


def md_table(columns, rows):
    return (["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def q(x):
    def s(a):
        return "never" if not np.isfinite(a) else f"{a:g}"
    v = [np.quantile(np.asarray(x, float), p) for p in (.5, .25, .75)]
    return f"{s(v[0])} [{s(v[1])}, {s(v[2])}]"


def crossing(d):
    """Median [IQR] time to 50% and 90% weight (never reached = infinite), and how many runs reach each."""
    t50 = np.where(d.t50 >= 0, d.t50, np.inf)
    t90 = np.where(d.t90 >= 0, d.t90, np.inf)
    return [q(t50), q(t90), f"{(d.t50 >= 0).sum()}/{len(d)} / {(d.t90 >= 0).sum()}/{len(d)}"]


def paired(dd):
    return dd.mean(), 2 * standard_error(dd)


def lost_majority(d):
    return f"{int(d.incorrect_majority.sum())}/{len(d)}"


def identification():
    initial, switch = [], []
    for g, gl in CONSTANT_FAMILIES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == g) & (p.t >= C.BURN)]["mean"].mean()
        initial.append([gl, *crossing(d), lost_majority(d),
                        f"{d.share_ge50_after.median():.3f} ({d.share_ge50_after.min():.3f})", f"{w:.3f}"])
    for g, gl in SWITCHES:
        d, p = read_family_results("ident", g), read_family_results("path", g)
        w = p[(p.family == d.target.iloc[0]) & (p.t >= SETTLED_START)]["mean"].mean()
        switch.append([gl, *crossing(d), f"{d.pre_weight_target.mean():.4f}", f"{w:.3f}", lost_majority(d)])
    return (["## Initial identification (constant family, uniform start)\n"]
            + md_table(["Family", "50%, median [IQR]", "90%", "reached 50 / 90%", "lost majority", "≥ 50% after (median, min)",
                        f"weight {EVALUATION_TICKS}"], initial)
            + ["\n## Switch adaptation (observations after the switch)\n"]
            + md_table(["Switch", "50%, median [IQR]", "90%", "reached 50 / 90%", "weight before", f"weight {SETTLED_TICKS}",
                        "lost majority"], switch))


def density(rows):
    switch, constant = [], []
    for g, gl in SWITCHES:
        x = paired_methods(read_family_results("seeds", g), "excess_x1000", ["method", "window"])
        cells = []
        for w in ["rec_0_100", "late", "post", "pre"]:
            m, se2 = paired(x[("gaussian_mff", w)] - x[("family_mff", w)])
            rows.append(dict(regime=g, window=w, dL=m, dL_2se=se2, n=len(x)))
            cells.append(format_estimate(m, se2, "+.1f", ".1f"))
        switch.append([gl, *cells, f"{x[('gaussian_mff', 'post')].mean():.2f} / {x[('family_mff', 'post')].mean():.2f}"])
    for g, gl in CONSTANT_FAMILIES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "excess_x1000")
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        rows.append(dict(regime=g + "_constant", window="whole", dL=m, dL_2se=se2, n=len(x)))
        constant.append([gl, format_estimate(m, se2), f"{x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f}"])
    return (["\n## Density accuracy: ΔL = Gaussian-only − family (×10⁻³ nats/obs, mean ± 2 SE)\n"]
            + md_table(["Switch", f"early {EARLY_TICKS}", f"later {LATE_TICKS}", f"post {POST_TICKS}", f"pre {PRE_TICKS}",
                        "L Gauss / family (post)"], switch)
            + [f"\nConstant families ({EVALUATION_TICKS}); negative = cost of augmentation\n"]
            + md_table(["Family", "ΔL ± 2 SE", "L Gauss / family"], constant))


def rmse():
    rows = []
    for g, gl in CONSTANT_FAMILIES + SWITCHES:
        s = read_family_results("seeds", g)
        x = paired_methods(s[s.window == "whole"], "rmse") * 100
        m, se2 = paired(x.gaussian_mff - x.family_mff)
        rows.append([gl, f"{x.gaussian_mff.mean():.2f} / {x.family_mff.mean():.2f}", format_estimate(m, se2)])
    guard = pd.concat([read_family_results("ident", g) for g, _ in CONSTANT_FAMILIES + SWITCHES]).n_guard.sum()
    return (["\n## ρ_S RMSE ×10⁻², whole run (positive difference favours family)\n"]
            + md_table(["Scenario", "Gauss / family", "difference"], rows)
            + [f"\nNon-finite model log densities replaced by Mff's guard: {int(guard)}."])


def accuracy():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (e, _), (g, _) in itertools.product(ESTIMATORS, SWITCHES):
        s = read_family_results(f"seeds_h2_{e}", g)
        for w, (m, _) in itertools.product(["post", "rec_0_100", "late"], ACCURACY_METRICS):
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
    post = D[D.window == "post"]
    main = []
    for (m, ml), (g, gl), (e, el) in itertools.product(ACCURACY_METRICS, SWITCHES, ESTIMATORS):
        r = row(D, e, g, "post", m)
        main.append([ml, gl, el, f"{r.L_gauss:.2f}", f"{r.L_family:.2f}", format_estimate(r.dL, r.dL_2se),
                     f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}]"])
    windows = [[gl, ml] + [format_estimate(r.dL, r.dL_2se) for r in (row(D, "rmpl", g, w, m) for w in ("rec_0_100", "late", "post"))]
               for (g, gl), (m, ml) in itertools.product(SWITCHES, ACCURACY_METRICS)]
    spread = []
    for (m, ml), (g, gl) in itertools.product(ACCURACY_METRICS, SWITCHES):
        x = post[(post.regime == g) & (post.metric == m)]
        spread.append([ml, gl, f"{x.pct.max() - x.pct.min():.2f}", f"{x.pct_2se.max():.2f}"])
    smallest = {m: post[post.metric == m].min_gauss.min() for m, _ in ACCURACY_METRICS}
    return (["# Section 6.5 accuracy across estimators: family-augmented vs Gaussian-only MFF\n",
             f"Post-switch ({POST_TICKS}), ×10⁻³. % = 100 (L_G − L_F)/L_G, ± 2 SE and 95% CI by paired-seed bootstrap.\n"]
            + md_table(["Metric", "Switch", "Estimator", "Gauss", "Family", "reduction ± 2 SE", "% ± 2 SE [95% CI]"], main)
            + [f"\nSmallest per-seed Gaussian-only value: {smallest['excess_x1000']:.2f} (excess) / {smallest['h2_x1000']:.2f} (H²) ×10⁻³.",
               "\n## RMPL recovery windows (reduction ×10⁻³, mean ± 2 SE)\n"]
            + md_table(["Switch", "Metric", f"early {EARLY_TICKS}", f"later {LATE_TICKS}", "post"], windows)
            + [f"\nH² checkpoints every 10 obs. in {EARLY_TICKS}, every 50 elsewhere.", "\n## Estimator agreement after the switch\n"]
            + md_table(["Metric", "Switch", "spread (pp)", "largest 2 SE (pp)"], spread))


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
    write_rmpl()
    write_accuracy()


if __name__ == "__main__":
    main()
