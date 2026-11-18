"""Section 6.3 accuracy of fixed forgetting (hindsight-best memory per cell), VFF and MFF, and sensitivity to the VFF
learning rate: RMSE, excess log loss and squared Hellinger per family, estimator, scenario and window, with paired
differences. Writes results/strategies_summary.csv, results/vffsens_*.csv and, with REPORTS=1, the text reports
(DENS_TAG=mix: the mixture-density run)."""
import itertools
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error as se  # noqa: E402
from summary_stats import win_loss_tie  # noqa: E402
import config as C  # noqa: E402

R = "results/"
TAG = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""
FIXED_ARMS = [f"fixed_{m}" for m in C.MEMORIES]
SUMMARY_STRATEGIES = ["fixed", "vff_slow", "vff_fast", "mff"]
RATES = [f"{r:g}" for r in C.VFF_RATES]
DEV_RATE = "0.02"
MEAN4 = "mean4"
METRICS = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
           ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]
SCENARIO_WINDOWS = [(sc, w) for sc in C.SCENARIOS for w in C.WINDOWS[sc]]
STRATEGY_COLUMNS = (f"{'family':9s} {'estimator':16s} {'Fixed (N)':>14s} {'VFF slow':>12s} "
                    f"{'VFF fast':>12s} {'MFF':>12s} {'MFF-Fixed':>14s} {'MFF-VFFslow':>14s}")


def wtl(diff):
    return win_loss_tie(diff.mean(), se(diff))


def pm(diff):
    return f"{diff.mean():+7.2f}±{2 * se(diff):.2f}"


def per_seed(seeds, family, scenario, estimator, window, metric, scale):
    """Seeds x strategies for one cell, and the fixed memory with the lowest mean (the hindsight-best fixed arm)."""
    cell = seeds[(seeds.copula == family) & (seeds.scenario == scenario) & (seeds.estimator == estimator) & (seeds.window == window)]
    by_strategy = cell.pivot(index="seed", columns="strategy", values=metric) * scale
    return by_strategy, min(FIXED_ARMS, key=lambda k: by_strategy[k].mean())


def strategy_cell(seeds, family, scenario, estimator, window, metric, scale):
    by_strategy, best = per_seed(seeds, family, scenario, estimator, window, metric, scale)
    cell = pd.DataFrame({"fixed": by_strategy[best], "vff_slow": by_strategy["vff_slow"],
                         "vff_fast": by_strategy["vff_fast"], "mff": by_strategy["mff"]})
    return cell, best


def strategy_header():
    mff = "the mixture of its candidates' Gaussian densities" if TAG else "the Gaussian at its forecast"
    return [f"6.3 accuracy, 60 seeds. Density: fixed and VFF use the Gaussian at their forecast, MFF {mff}.",
            "Fixed = hindsight-best memory per cell and metric. Differences paired, mean ± 2 SE."]


def strategy_line(label, estimator, cell, best=None):
    """One table row; best is the hindsight memory, or None for the family mean (which adds MFF - VFF fast)."""
    mean, err = cell.mean(), cell.apply(se)
    fixed = f"{mean.fixed:7.2f}" + (f"({best[6:]:>4s})" if best else " " * 6)
    line = (f"{label:9s} {C.EST_LABEL[estimator]:16s} {fixed} {mean.vff_slow:7.2f}({err.vff_slow:.2f}) "
            f"{mean.vff_fast:7.2f}({err.vff_fast:.2f}) {mean.mff:7.2f}({err.mff:.2f}) "
            f"{pm(cell.mff - cell.fixed)} {pm(cell.mff - cell.vff_slow)}")
    if best is None:
        fast = cell.mff - cell.vff_fast
        line += f"   (MFF-VFFfast {fast.mean():+.2f}±{2 * se(fast):.2f})"
    return line


def summary_rows(key, cell, best):
    mean, err = cell.mean(), cell.apply(se)
    return [key | dict(strategy=k, mean=mean[k], se=err[k], best_fixed=best) for k in SUMMARY_STRATEGIES]


def strategy_tables(seeds):
    out, rows = [], []
    for metric, label, scale in METRICS:
        out += ["", label, STRATEGY_COLUMNS]
        for scenario, window in SCENARIO_WINDOWS:
            out += ["", f"{scenario} / {window}"]
            for estimator in C.ESTIMATORS:
                key = dict(metric=metric, scenario=scenario, window=window, estimator=estimator)
                cells = {f: strategy_cell(seeds, f, scenario, estimator, window, metric, scale) for f in C.FAMILIES}
                for family, (cell, best) in cells.items():
                    out.append(strategy_line(C.FAMILY_TEXT[family], estimator, cell, best))
                    rows += summary_rows(key | {"copula": family}, cell, best)
                family_mean = sum(cell for cell, _ in cells.values()) / len(cells)
                out.append(strategy_line("mean of 4", estimator, family_mean))
                rows += summary_rows(key | {"copula": MEAN4}, family_mean, "")
    return out, pd.DataFrame(rows)


def agreement(seeds):
    """W/T/L of MFF against hindsight fixed for each estimator x family, one letter per metric."""
    def code(scenario, estimator, family):
        cells = (strategy_cell(seeds, family, scenario, estimator, "whole", m, scale)[0] for m, _, scale in METRICS)
        return "".join(wtl(cell.mff - cell.fixed) for cell in cells)
    out = ["", "MFF vs hindsight fixed by metric (whole run, W = lower loss at 2 SE)"]
    for scenario in C.SCENARIOS:
        codes = [code(scenario, e, f) for e, f in itertools.product(C.ESTIMATORS, C.FAMILIES)]
        out.append(f"{scenario:9s} (RMSE/excess/H2 per family x estimator): " + " ".join(codes))
    return out


def strategies():
    seeds = pd.concat([pd.read_csv(R + f"strategies_seeds{TAG}{C.sfx(f)}.csv") for f in C.FAMILIES])
    body, summary = strategy_tables(seeds)
    summary.to_csv(R + f"strategies_summary{TAG}.csv", index=False)
    if reports_enabled():
        lines = strategy_header() + body + agreement(seeds)
        write_report(R + ("strategies_2026-11-12.txt" if not TAG else f"strategies{TAG}_2026-11-14.txt"), lines)


def load_vffsens(kind):
    return pd.concat([pd.read_csv(R + f"strategies_{kind}_vffsens{TAG[1:]}{C.sfx(f)}.csv") for f in C.FAMILIES])


def rate_table(seeds, metric, scale, scenario, window, estimator, family):
    """Seeds x (VFF at each alpha, MFF, hindsight fixed); family may be MEAN4, the per-seed mean over the four families."""
    def one(f):
        by_strategy, best = per_seed(seeds, f, scenario, estimator, window, metric, scale)
        table = pd.DataFrame({a: by_strategy[f"vff_{a}"] for a in RATES})
        table["mff"] = by_strategy["mff"]
        table["fixed"] = by_strategy[best]
        return table
    return sum(one(f) for f in C.FAMILIES) / 4 if family == MEAN4 else one(family)


def rate_summary(seeds):
    rows = []
    cells = itertools.product(METRICS, SCENARIO_WINDOWS, C.ESTIMATORS, C.FAMILIES + [MEAN4])
    for (metric, _, scale), (scenario, window), estimator, family in cells:
        table = rate_table(seeds, metric, scale, scenario, window, estimator, family)
        key = dict(metric=metric, scenario=scenario, window=window, estimator=estimator, copula=family)
        for a in RATES:
            diff = table.mff - table[a]
            rows.append(key | dict(strategy=f"vff_{a}", mean=table[a].mean(), se=se(table[a]),
                                   mff_minus=diff.mean(), mff_minus_se=se(diff), verdict=wtl(diff)))
        for k in ("mff", "fixed"):
            rows.append(key | dict(strategy=k, mean=table[k].mean(), se=se(table[k])))
    return pd.DataFrame(rows)


def summary_cell(summary, metric, scenario, window, estimator):
    return summary[(summary.metric == metric) & (summary.scenario == scenario) & (summary.window == window)
                   & (summary.estimator == estimator) & (summary.copula == MEAN4)].set_index("strategy")


def rate_lines(summary, metric, scenario, window, estimator):
    """Family-mean VFF at each alpha, MFF and fixed; then MFF - VFF with its verdict and 2 SE."""
    cell = summary_cell(summary, metric, scenario, window, estimator)
    vff = [cell.loc["vff_" + a] for a in RATES]
    return [f"{C.EST_LABEL[estimator]:16s}   " + " ".join(f"{v['mean']:7.2f}" for v in vff)
            + f" | {cell.loc['mff', 'mean']:7.2f} {cell.loc['fixed', 'mean']:7.2f}",
            f"{'':16s}   " + " ".join(f"{v.mff_minus:+6.2f}{v.verdict}" for v in vff),
            f"{'':16s}   " + " ".join(f"±{2 * v.mff_minus_se:5.2f} " for v in vff)]


def sensitivity(summary):
    out = ["VFF learning-rate sensitivity, 60 seeds, MFF eta = 1, delta = 1e-4. alpha = 0.02 is the development choice.",
           "MFF − VFF paired, mean ± 2 SE; W/T/L = MFF lower / within 2 SE / higher."]
    columns = f"{'Estimator':16s} a=" + " ".join(f"{a:>7s}" for a in RATES) + f" | {'MFF':>7s} {'Fixed':>7s}"
    for metric, label, _ in METRICS:
        out += ["", label, columns]
        for scenario, window in SCENARIO_WINDOWS:
            out += ["", f"{scenario} / {window}"]
            for estimator in C.ESTIMATORS:
                out += rate_lines(summary, metric, scenario, window, estimator)
    return out


def best_alpha(summary):
    """Hindsight-best alpha per scenario, and the single alpha minimising the mean ratio to the per-scenario best."""
    out, rows = ["", "Hindsight-best alpha (whole run, family mean)"], []
    for metric, label, _ in METRICS:
        out.append(f"-- {label}")
        for estimator in C.ESTIMATORS:
            means = {sc: summary_cell(summary, metric, sc, "whole", estimator)["mean"][[f"vff_{a}" for a in RATES]]
                     for sc in C.SCENARIOS}
            cells = []
            for scenario, m in means.items():
                alpha = m.idxmin()[4:]
                cells.append(f"{scenario}: {alpha:>5s} ({m.min():.2f})")
                rows.append(dict(metric=metric, estimator=estimator, scenario=scenario, best_alpha=alpha, value=m.min()))
            ratios = {a: [m["vff_" + a] / m.min() for m in means.values()] for a in RATES}
            ratio = {a: (np.mean(r), np.max(r)) for a, r in ratios.items()}
            single = min(ratio, key=lambda a: ratio[a][0])
            rows.append(dict(metric=metric, estimator=estimator, scenario="all (single alpha)", best_alpha=single, value=ratio[single][0]))
            others = ", ".join(f"{a}: {ratio[a][0]:.3f}/{ratio[a][1]:.3f}" for a in ("0.05", "0.075", "0.02"))
            out.append(f"   {C.EST_LABEL[estimator]:16s} " + "   ".join(cells)
                       + f"   | best single alpha: {single} (mean ratio to per-scenario best {ratio[single][0]:.3f}, "
                       f"worst {ratio[single][1]:.3f}); " + others)
    return out, pd.DataFrame(rows)


def comparisons(seeds, best):
    """MFF - VFF at the development alpha, the per-scenario best alpha and the best single alpha."""
    def chosen(metric, estimator, scenario):
        return best[(best.metric == metric) & (best.estimator == estimator) & (best.scenario == scenario)].best_alpha.iloc[0]
    out = ["", "MFF - VFF (whole run, family mean ± 2 SE)"]
    for metric, label, scale in METRICS:
        out.append(f"-- {label}")
        for estimator in C.ESTIMATORS:
            single = chosen(metric, estimator, "all (single alpha)")
            out.append(f"   {C.EST_LABEL[estimator]}:")
            for scenario in C.SCENARIOS:
                best_here = chosen(metric, estimator, scenario)
                table = rate_table(seeds, metric, scale, scenario, "whole", estimator, MEAN4)
                parts = []
                for name, alpha in [("0.02(dev)", DEV_RATE), (f"best {best_here}", best_here), (f"single {single}", single)]:
                    diff = table.mff - table[alpha]
                    parts.append(f"{name} {diff.mean():+.2f}±{2 * se(diff):.2f}{wtl(diff)}")
                every = "".join(wtl(table.mff - table[a]) for a in RATES)
                out.append(f"      {scenario}: " + ", ".join(parts) + f", every alpha {every}")
    return out


def robustness(seeds):
    """Per family, MFF against that family's own hindsight-best alpha, counted as W/T/L."""
    out = ["", "MFF vs hindsight-best alpha per family (whole run)"]
    for metric, label, scale in METRICS:
        verdicts = {sc: [] for sc in C.SCENARIOS}
        for scenario, estimator, family in itertools.product(C.SCENARIOS, C.ESTIMATORS, C.FAMILIES):
            table = rate_table(seeds, metric, scale, scenario, "whole", estimator, family)
            alpha = min(RATES, key=lambda a: table[a].mean())
            verdicts[scenario].append(wtl(table.mff - table[alpha]))
        counts = "  ".join(f"{sc} {v.count('W')}/{v.count('T')}/{v.count('L')}" for sc, v in verdicts.items())
        out.append(f"{label} (W/T/L)  {counts}")
    return out


def extremes(tails):
    out = ["", "Excess log loss extremes (untrimmed): seeds and seed-ticks with |rho_hat| > 0.98, summed over families;",
           "top-1% = share of the excess from the worst 1% of seed-ticks."]
    for estimator, scenario in itertools.product(C.ESTIMATORS, C.SCENARIOS):
        cell = tails[(tails.scenario == scenario) & (tails.estimator == estimator)]
        out.append(f"-- {C.EST_LABEL[estimator]}, {scenario}")
        for strategy in [f"vff_{a}" for a in RATES] + ["mff"]:
            s = cell[cell.strategy == strategy]
            out.append(f"   {strategy:10s} seeds {int(s.seeds_extreme.sum()):3d}  seed-ticks {int(s.seedticks_extreme.sum()):5d}  max {s.ex_max.max():9.0f}"
                       f"  mean {s.ex_mean.mean():8.2f}  median {s.ex_median.mean():6.2f}  top-1% share {s.top1pct_share.mean():5.2f}")
    return out


def vffsens():
    seeds = load_vffsens("seeds")
    summary = rate_summary(seeds)
    summary.to_csv(R + f"vffsens_summary{TAG}.csv", index=False)
    best_lines, best = best_alpha(summary)
    best.to_csv(R + f"vffsens_best_alpha{TAG}.csv", index=False)
    tails = load_vffsens("extremes")
    tails.to_csv(R + f"vffsens_extremes{TAG}.csv", index=False)
    if reports_enabled():
        lines = sensitivity(summary) + best_lines + comparisons(seeds, best) + robustness(seeds) + extremes(tails)
        write_report(R + ("vffsens_2026-11-12.txt" if not TAG else f"vffsens{TAG}_2026-11-14.txt"), lines)


def main():
    strategies()
    vffsens()


if __name__ == "__main__":
    main()
