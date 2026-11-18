"""Section 6.5 tables: family identification, adaptation after a switch, and predictive accuracy of family-augmented
against Gaussian-only MFF."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import config as C  # noqa: E402
from report_output import format_estimate, md_table, reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402

RESULTS = Path("results")
CONSTANT_FAMILIES = [(f, C.FAMILY_LONG_TEXT[f]) for f in C.FAMILIES_T31]
SWITCHES = [(g, f"{C.FAMILY_TEXT[a]} → {C.FAMILY_TEXT[b]}") for g, a, b in C.SWITCHES]
ESTIMATORS = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
ACCURACY_METRICS = [("excess_x1000", "Excess log loss"), ("h2_x1000", "Squared Hellinger")]
DENSITY_WINDOWS = ["rec_0_100", "late", "post", "pre"]
ACCURACY_WINDOWS = ["post", "rec_0_100", "late"]
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


def read_results(kind, regime):
    return pd.read_csv(RESULTS / f"famsw_{kind}_{regime}.csv")


def by_method(rows, metric, columns="method"):
    return rows.pivot_table(index="seed", columns=columns, values=metric)


def whole_run(regime, metric):
    seeds = read_results("seeds", regime)
    return by_method(seeds[seeds.window == "whole"], metric)


def median_iqr(times):
    def show(t):
        return "never" if not np.isfinite(t) else f"{t:g}"
    median, q1, q3 = (np.quantile(np.asarray(times, float), p) for p in (.5, .25, .75))
    return f"{show(median)} [{show(q1)}, {show(q3)}]"


def crossing(ident):
    """Median [IQR] time to 50% and 90% weight (never reached = infinite), and how many runs reach each."""
    t50 = np.where(ident.t50 >= 0, ident.t50, np.inf)
    t90 = np.where(ident.t90 >= 0, ident.t90, np.inf)
    n = len(ident)
    return [median_iqr(t50), median_iqr(t90), f"{(ident.t50 >= 0).sum()}/{n} / {(ident.t90 >= 0).sum()}/{n}"]


def paired(diff):
    return diff.mean(), 2 * standard_error(diff)


def lost_majority(ident):
    return f"{int(ident.incorrect_majority.sum())}/{len(ident)}"


def identification():
    initial, switch = [], []
    for regime, label in CONSTANT_FAMILIES:
        ident, path = read_results("ident", regime), read_results("path", regime)
        weight = path[(path.family == regime) & (path.t >= C.BURN)]["mean"].mean()
        share = ident.share_ge50_after
        initial.append([label, *crossing(ident), lost_majority(ident), f"{share.median():.3f} ({share.min():.3f})", f"{weight:.3f}"])
    for regime, label in SWITCHES:
        ident, path = read_results("ident", regime), read_results("path", regime)
        weight = path[(path.family == ident.target.iloc[0]) & (path.t >= SETTLED_START)]["mean"].mean()
        switch.append([label, *crossing(ident), f"{ident.pre_weight_target.mean():.4f}", f"{weight:.3f}", lost_majority(ident)])
    return (["## Initial identification (constant family, uniform start)\n"]
            + md_table(["Family", "50%, median [IQR]", "90%", "reached 50 / 90%", "lost majority", "≥ 50% after (median, min)",
                        f"weight {EVALUATION_TICKS}"], initial)
            + ["\n## Switch adaptation (observations after the switch)\n"]
            + md_table(["Switch", "50%, median [IQR]", "90%", "reached 50 / 90%", "weight before", f"weight {SETTLED_TICKS}",
                        "lost majority"], switch))


def density(rows):
    switch, constant = [], []
    for regime, label in SWITCHES:
        loss = by_method(read_results("seeds", regime), "excess_x1000", ["method", "window"])
        cells = []
        for window in DENSITY_WINDOWS:
            diff, err = paired(loss[("gaussian_mff", window)] - loss[("family_mff", window)])
            rows.append(dict(regime=regime, window=window, dL=diff, dL_2se=err, n=len(loss)))
            cells.append(format_estimate(diff, err, "+.1f", ".1f"))
        post = f"{loss[('gaussian_mff', 'post')].mean():.2f} / {loss[('family_mff', 'post')].mean():.2f}"
        switch.append([label, *cells, post])
    for regime, label in CONSTANT_FAMILIES:
        loss = whole_run(regime, "excess_x1000")
        diff, err = paired(loss.gaussian_mff - loss.family_mff)
        rows.append(dict(regime=regime + "_constant", window="whole", dL=diff, dL_2se=err, n=len(loss)))
        constant.append([label, format_estimate(diff, err), f"{loss.gaussian_mff.mean():.2f} / {loss.family_mff.mean():.2f}"])
    return (["\n## Density accuracy: ΔL = Gaussian-only − family (×10⁻³ nats/obs, mean ± 2 SE)\n"]
            + md_table(["Switch", f"early {EARLY_TICKS}", f"later {LATE_TICKS}", f"post {POST_TICKS}", f"pre {PRE_TICKS}",
                        "L Gauss / family (post)"], switch)
            + [f"\nConstant families ({EVALUATION_TICKS}); negative = cost of augmentation\n"]
            + md_table(["Family", "ΔL ± 2 SE", "L Gauss / family"], constant))


def rmse():
    rows = []
    for regime, label in CONSTANT_FAMILIES + SWITCHES:
        error = whole_run(regime, "rmse") * 100
        diff, err = paired(error.gaussian_mff - error.family_mff)
        rows.append([label, f"{error.gaussian_mff.mean():.2f} / {error.family_mff.mean():.2f}", format_estimate(diff, err)])
    guard = pd.concat([read_results("ident", regime) for regime, _ in CONSTANT_FAMILIES + SWITCHES]).n_guard.sum()
    return (["\n## ρ_S RMSE ×10⁻², whole run (positive difference favours family)\n"]
            + md_table(["Scenario", "Gauss / family", "difference"], rows)
            + [f"\nNon-finite model log densities replaced by Mff's guard: {int(guard)}."])


def accuracy():
    """Per estimator, switch, window and metric: losses, the paired reduction and the bootstrap percentage reduction."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (estimator, _), (regime, _) in itertools.product(ESTIMATORS, SWITCHES):
        seeds = read_results(f"seeds_h2_{estimator}", regime)
        for window, (metric, _) in itertools.product(ACCURACY_WINDOWS, ACCURACY_METRICS):
            loss = by_method(seeds[seeds.window == window], metric)
            gauss, family = loss.gaussian_mff.values, loss.family_mff.values
            diff = gauss - family
            draws = rng.integers(0, len(gauss), size=(BOOTSTRAP_DRAWS, len(gauss)))
            boot_pct = 100 * (gauss[draws].mean(1) - family[draws].mean(1)) / gauss[draws].mean(1)
            rows.append(dict(estimator=estimator, regime=regime, window=window, metric=metric, L_gauss=gauss.mean(),
                             L_family=family.mean(), dL=diff.mean(), dL_2se=2 * standard_error(diff),
                             pct=100 * (gauss.mean() - family.mean()) / gauss.mean(), pct_2se=2 * boot_pct.std(ddof=1),
                             pct_lo=np.percentile(boot_pct, 2.5), pct_hi=np.percentile(boot_pct, 97.5),
                             min_gauss=gauss.min(), n=len(gauss)))
    return pd.DataFrame(rows)


def accuracy_row(results, estimator, regime, window, metric):
    return results[(results.estimator == estimator) & (results.regime == regime) & (results.window == window)
                   & (results.metric == metric)].iloc[0]


def accuracy_report(results):
    post = results[results.window == "post"]
    main = []
    for (metric, mlabel), (regime, slabel), (estimator, elabel) in itertools.product(ACCURACY_METRICS, SWITCHES, ESTIMATORS):
        r = accuracy_row(results, estimator, regime, "post", metric)
        main.append([mlabel, slabel, elabel, f"{r.L_gauss:.2f}", f"{r.L_family:.2f}", format_estimate(r.dL, r.dL_2se),
                     f"{r.pct:.1f} ± {r.pct_2se:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}]"])
    windows = []
    for (regime, slabel), (metric, mlabel) in itertools.product(SWITCHES, ACCURACY_METRICS):
        cells = [accuracy_row(results, "rmpl", regime, w, metric) for w in ("rec_0_100", "late", "post")]
        windows.append([slabel, mlabel] + [format_estimate(r.dL, r.dL_2se) for r in cells])
    spread = []
    for (metric, mlabel), (regime, slabel) in itertools.product(ACCURACY_METRICS, SWITCHES):
        cell = post[(post.regime == regime) & (post.metric == metric)]
        spread.append([mlabel, slabel, f"{cell.pct.max() - cell.pct.min():.2f}", f"{cell.pct_2se.max():.2f}"])
    smallest = {metric: post[post.metric == metric].min_gauss.min() for metric, _ in ACCURACY_METRICS}
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
