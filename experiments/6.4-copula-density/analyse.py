"""Section 6.4 tables: squared vs log loss across generating and working copulas, and MFF regret. Writes
results/scoresfam_cells.csv and, with REPORTS=1, the scoring and regret reports."""
import itertools
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import format_estimate, reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402

R = "results/"
FAMILIES = C.FAMILIES_T31
FAMILY_SHORT = {"gaussian": "Gauss", "t": "t4", "clayton": "Clay", "gumbel": "Gumb"}
ESTIMATOR_INV = {"rmpl": "RMPL (inv)", "xiao": "Counting Matrix (inv)", "hermite": "Hermite (inv)"}
METRICS = [("rmse", "RMSE x1e-2", 100), ("h2_x1000", "H2 x1e-3", 1), ("excess_x1000", "excess x1e-3", 1)]
VERDICTS = ("log", "tie", "sq")
JUMP_WINDOWS = ["pre", "rec_0_100", "rec_100_500", "post"]
CELL_KEY = ["gen", "scenario", "estimator", "working", "window"]
WORKING_LABEL = {f: "RMPL " + C.FAMILY_TEXT[f] for f in FAMILIES}
ORACLES = ("best0", "best1", "best2")
N_ARMS = len(C.MEMORIES)
HEDGE, FIXED_SHARE = 0.0, 1e-4


def load_scoresfam(tag):
    seeds = pd.concat([pd.read_csv(R + f"scoresfam_{tag}{C.sfx(f)}.csv") for f in FAMILIES])
    return seeds[seeds.estimator.isin(C.ESTIMATORS)]


def paired_cells(seeds):
    """Per cell and metric: mean and median under each score, and squared - log paired over seeds."""
    squared = seeds[seeds.score == "squared"].set_index(CELL_KEY + ["seed"])
    log = seeds[seeds.score == "log"].set_index(CELL_KEY + ["seed"])
    rows = []
    for key, _ in squared.groupby(level=list(range(len(CELL_KEY)))):
        sq_cell, log_cell = squared.loc[key], log.loc[key]
        for metric, _, scale in METRICS:
            sq_values, log_values = sq_cell[metric] * scale, log_cell[metric] * scale
            diff = (sq_values - log_values).dropna()
            mean, se = diff.mean(), standard_error(diff)
            rows.append(dict(zip(CELL_KEY, key), metric=metric, sq=sq_values.mean(), log=log_values.mean(),
                             sq_med=sq_values.median(), log_med=log_values.median(), diff=mean, se=se, diff_med=diff.median(),
                             verdict="log" if mean > 2 * se else "sq" if mean < -2 * se else "tie"))
    return pd.DataFrame(rows)


def format_diff(cell):
    estimate = format_estimate(cell["diff"], 2 * cell["se"], "+7.3f", ".3f", separator="±")
    return estimate + ("*" if cell["verdict"] != "tie" else " ")


def tally_verdicts(rows):
    return rows.verdict.value_counts().reindex(VERDICTS, fill_value=0)


def whole_run(cells, gen, scenario, estimator, working, metric):
    return cells[(cells.gen == gen) & (cells.scenario == scenario) & (cells.estimator == estimator) & (cells.working == working)
                 & (cells.window == "whole") & (cells.metric == metric)].iloc[0]


def gaussian_table(cells):
    """Δ for the Gaussian working family, by generating family and scenario."""
    out = ["", "Gaussian working family by generating family (Δ)"]
    for estimator in C.ESTIMATORS:
        out += [f"-- {ESTIMATOR_INV[estimator]}", f"{'gen':6s} {'scenario':9s} " + " ".join(f"{label:>17s}" for _, label, _ in METRICS)]
        for gen, scenario in itertools.product(FAMILIES, C.SCENARIOS):
            diffs = [format_diff(whole_run(cells, gen, scenario, estimator, "gaussian", m)) for m, _, _ in METRICS]
            out.append(f"{FAMILY_SHORT[gen]:6s} {scenario:9s} " + " ".join(f"{d:>17s}" for d in diffs))
    return out


def family_grid(cells, title, width, show):
    """Generating x working family grid per metric, scenario and estimator; show(cell, matched) formats one entry."""
    out = ["", title]
    header = f"   {'gen \\ W':8s} " + " ".join(f"{FAMILY_SHORT[w]:>{width}s}" for w in FAMILIES)
    for metric, label, _ in METRICS:
        for scenario in C.SCENARIOS:
            out.append(f"-- {label}, {scenario}")
            for estimator in C.ESTIMATORS:
                out += [f"   {ESTIMATOR_INV[estimator]}", header]
                for gen in FAMILIES:
                    row = [show(whole_run(cells, gen, scenario, estimator, w, metric), gen == w) for w in FAMILIES]
                    out.append(f"   {FAMILY_SHORT[gen]:8s} " + " ".join(f"{c:>{width}s}" for c in row))
    return out


def verdict_counts(cells):
    """How often log loss, squared loss or neither wins, overall and by estimator, specification and jump window."""
    whole = cells[cells.window == "whole"]

    def by_metric(rows):
        return "; ".join(f"{label} " + "/".join(map(str, tally_verdicts(rows[rows.metric == m]))) for m, label, _ in METRICS)
    out = ["", "Whole-run verdicts (log / tie / squared)"]
    for metric, label, _ in METRICS:
        rows = whole[whole.metric == metric]
        log, tie, sq = tally_verdicts(rows)
        out.append(f"{label:14s} log {log:3d}  tie {tie:3d}  sq {sq:3d}   (of {len(rows)})")
    out.append("by estimator:")
    out += ["  " + ESTIMATOR_INV[e] + ": " + by_metric(whole[whole.estimator == e]) for e in C.ESTIMATORS]
    out.append("matched (diagonal) vs misspecified (off-diagonal):")
    out += ["  " + label + ": " + by_metric(whole[mask]) for label, mask in
            [("matched", whole.gen == whole.working), ("misspecified", whole.gen != whole.working)]]
    out.append("squared-loss wins (all windows):")
    squared_wins = cells[cells.verdict == "sq"]
    columns = CELL_KEY + ["metric", "sq", "log", "diff", "se", "diff_med"]
    out.append(squared_wins[columns].round(3).to_string(index=False) if len(squared_wins) else "none")
    out += ["", "Jump-window verdicts (log / tie / squared)"]
    out += [f"{window:12s} " + by_metric(cells[cells.window == window]) for window in JUMP_WINDOWS]
    out += ["", "RMSE vs density verdict reversals"]
    by_cell = whole.pivot_table(index=CELL_KEY[:4], columns="metric", values="verdict", aggfunc="first")
    reversals = by_cell[(by_cell.rmse != by_cell.h2_x1000) | (by_cell.rmse != by_cell.excess_x1000)]
    return out + [f"{len(reversals)} of {len(by_cell)} cells", reversals.to_string() if len(reversals) else ""]


def matching(seeds):
    """Log-loss MFF with the matched working family minus with the Gaussian one, paired over seeds."""
    out = ["", "Log-loss MFF (matched minus Gaussian working family)"]
    for estimator, (metric, label, _) in itertools.product(C.ESTIMATORS, METRICS):
        scale = 100 if metric == "rmse" else 1
        cells = []
        for gen, scenario in itertools.product(FAMILIES[1:], C.SCENARIOS):
            log = seeds[(seeds.gen == gen) & (seeds.scenario == scenario) & (seeds.estimator == estimator)
                        & (seeds.window == "whole") & (seeds.score == "log")]
            matched = log[log.working == gen].set_index("seed")[metric]
            gaussian = log[log.working == "gaussian"].set_index("seed")[metric]
            diff = (matched - gaussian) * scale
            estimate = format_estimate(diff.mean(), standard_error(diff, scale=2), "+.3f", ".3f", separator="±")
            cells.append(f"{FAMILY_SHORT[gen]}/{scenario[:5]} {estimate}")
        out.append(f"{ESTIMATOR_INV[estimator]:22s} {label:12s} " + "  ".join(cells))
    return out


def diagnostics(diag):
    columns = ["arm_clamp_ticks", "score_nonfinite", "eval_nonfinite"]
    return ["", "Numerical diagnostics (clamps from t = 500)",
            diag.groupby(["estimator", "working"])[columns].agg(["sum", "max"]).to_string()]


def eta_sensitivity(dev):
    """Each score at its own dev-best eta, against the fixed eta = 1, on the dev seeds."""
    dev = dev[dev.window == "whole"]
    metrics = ["rmse", "excess_x1000"]
    means = dev.groupby(CELL_KEY[:4] + ["score", "eta"])[metrics].mean().reset_index()
    rows = []
    for (gen, scenario, estimator, working), x in means.groupby(CELL_KEY[:4]):
        for metric in metrics:
            squared = x[x.score == "squared"].set_index("eta")[metric]
            log = x[x.score == "log"].set_index("eta")[metric]
            rows.append(dict(gen=gen, scenario=scenario, estimator=estimator, working=working, metric=metric,
                             eta_sq=squared.idxmin(), eta_log=log.idxmin(),
                             d_tuned=squared.min() - log.min(), d_eta1=squared.loc[1.0] - log.loc[1.0]))
    best = pd.DataFrame(rows)
    best["rev"] = np.sign(best.d_tuned) != np.sign(best.d_eta1)
    out = ["", "Learning-rate sensitivity (dev seeds 1–30): best eta per score vs eta = 1"]
    for metric in metrics:
        b = best[best.metric == metric]
        eta_sq = b.eta_sq.value_counts().sort_index().to_dict()
        eta_log = b.eta_log.value_counts().sort_index().to_dict()
        out += [f"{metric}: tuned favours log in {int((b.d_tuned > 0).sum())}/{len(b)} cells (eta = 1: {int((b.d_eta1 > 0).sum())}); "
                f"sign reversals {int(b.rev.sum())}",
                f"   dev-best eta, squared: {eta_sq}   log: {eta_log}"]
    reversed_cells = best[best.rev]
    return out + ([reversed_cells.round(4).to_string(index=False)] if len(reversed_cells) else [])


def scoresfam():
    seeds = load_scoresfam("seeds_sfeval")
    cells = paired_cells(seeds)
    cells.to_csv(R + "scoresfam_cells.csv", index=False)
    if not reports_enabled():
        return
    lines = ["MFF scoring rule by family. Δ = squared − log, paired over 60 seeds, mean ± 2 SE, * beyond 2 SE; positive = log better."]
    lines += gaussian_table(cells)
    lines += family_grid(cells, "Generating × working family (Δ, [ ] = matched)", 17,
                         lambda cell, matched: f"[{format_diff(cell)}]" if matched else f" {format_diff(cell)} ")
    lines += family_grid(cells, "Mean accuracy (log / squared, * = matched)", 15,
                         lambda cell, matched: f"{cell['log']:6.3f}/{cell['sq']:6.3f}" + ("*" if matched else " "))
    lines += verdict_counts(cells) + matching(seeds) + diagnostics(load_scoresfam("diag_sfeval"))
    if os.path.exists(R + "scoresfam_seeds_sfdev.csv"):
        lines += eta_sensitivity(load_scoresfam("seeds_sfdev"))
    write_report(R + "scoresfam_2026-11-15.txt", lines)


def bound(switches, delta, ticks):
    """Chapter 5 regret bound against the best sequence with this many switches (Hedge: static only)."""
    if delta == 0:
        return math.log(N_ARMS) if switches == 0 else float("nan")
    return (math.log(N_ARMS) + (ticks - 1 - switches) * math.log(1 / (1 - delta + delta / N_ARMS))
            + switches * math.log(N_ARMS / delta))


def bound_summary(regrets, ticks, switches, delta):
    limits = np.array([bound(switches, delta, n) for n in ticks])
    if np.isnan(limits).all():
        return None
    regrets = regrets.to_numpy()
    return regrets.max(), limits.max(), int((regrets > limits + 1e-9).sum()), len(regrets)


def fixed_share_and_hedge(runs):
    """The two weight updates' runs, indexed by seed so they pair."""
    return (runs[np.isclose(runs.delta, FIXED_SHARE)].set_index("seed"),
            runs[np.isclose(runs.delta, HEDGE)].set_index("seed"))


def mean_median(x):
    estimate = format_estimate(x.mean(), 2 * standard_error(x), "7.2f", "5.2f")
    return f"{estimate} (med {x.median():6.2f})"


def regret_lines(runs):
    out = []
    for delta, name in [(HEDGE, "Hedge      "), (FIXED_SHARE, "fixed share")]:
        update = runs[np.isclose(runs.delta, delta)]
        for switches, oracle in enumerate(ORACLES):
            mix, point = update.loss_mix - update[oracle], update.loss_point - update[oracle]
            summary = bound_summary(mix, update["T"], switches, delta)
            violations = ""
            if summary is not None:
                _, limit, count, n = summary
                violations = f" viol {count}/{n} (bound {limit:.2f})"
            out.append(f"   {name} m={switches}: mix {mean_median(mix)}{violations} | point {mean_median(point)}")
    return out


def switch_line(runs):
    hedge = runs[runs.delta == 0]
    switched = hedge[hedge.sw_t > 0]
    counts = switched.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False)
    arms = ", ".join(f"{a:g}->{b:g} x{n}" for (a, b), n in counts.items())
    return (f"   best one-switch sequence: {len(switched)}/{len(hedge)} runs switch; switch time median {switched.sw_t.median():.0f} "
            f"(IQR {switched.sw_t.quantile(.25):.0f}-{switched.sw_t.quantile(.75):.0f}; true change 2500); arms: " + arms)


def gaussian_regret(runs):
    """Static and switching regret of Hedge and fixed share, Gaussian copula, against the Chapter 5 bounds."""
    out = ["MFF regret, Gaussian copula (nats, log loss, eta = 1). mean ± 2 SE (median); viol = runs above the Chapter 5 bound."]
    for estimator in C.ESTIMATORS:
        out += ["", f"{C.EST_LABEL[estimator]} (T = {sorted(runs[runs.estimator == estimator]['T'].unique())})"]
        for scenario in C.SCENARIOS:
            cell = runs[(runs.estimator == estimator) & (runs.scenario == scenario)]
            hedge = cell[cell.delta == 0]
            out.append(f"-- {scenario}: best fixed arm (mode) {cell.arm0.mode().iloc[0]:g}; "
                       f"best fixed - best one-switch loss {mean_median(hedge.best0 - hedge.best1)}")
            out += regret_lines(cell)
            fs, h = fixed_share_and_hedge(cell)
            out.append(f"   fixed share - Hedge (paired, = difference in cumulative MFF loss): mix {mean_median(fs.loss_mix - h.loss_mix)} "
                       f"| point {mean_median(fs.loss_point - h.loss_point)}")
            if scenario == "jump":
                out.append(switch_line(cell))
    return out


def plus_minus(x):
    return format_estimate(x.mean(), 2 * standard_error(x), "6.2f", "4.2f", separator="±")


def switch_desc(hedge):
    switched = hedge[hedge.sw_t > 0]
    if not len(switched):
        return "no switch"
    top = switched.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False).head(2)
    return f"{len(switched)}/60 switch, t~{switched.sw_t.median():.0f}, " + ",".join(f"{a:g}>{b:g}x{n}" for (a, b), n in top.items())


def family_table(runs):
    """Regret of fixed share and Hedge for every generating x working family pair."""
    out = ["MFF regret across families (nats), mean ± 2 SE. FS = fixed share, H = Hedge, sw1 = one-switch regret,",
           "gain1 = best fixed − best one-switch loss, [M] = matched."]
    header = (f"{'gen':9s} {'working':13s} {'stat FS':>11s} {'stat H':>11s} {'sw1 H':>11s} {'sw1 FS':>11s} {'FS - H':>12s} "
              f"{'gain1':>11s}  best fixed / one-switch")
    for scenario in C.SCENARIOS:
        out += ["", scenario, header]
        for gen, working in itertools.product(FAMILIES, FAMILIES):
            fs, h = fixed_share_and_hedge(runs[(runs.gen == gen) & (runs.scenario == scenario) & (runs.working == working)])
            label = WORKING_LABEL[working] + (" [M]" if gen == working else "")
            cells = [plus_minus(fs.loss_mix - fs.best0), plus_minus(h.loss_mix - h.best0), plus_minus(h.loss_mix - h.best1),
                     plus_minus(fs.loss_mix - fs.best1)]
            out.append(f"{C.FAMILY_TEXT[gen]:9s} {label:13s} " + " ".join(f"{c:>11s}" for c in cells)
                       + f" {plus_minus(fs.loss_mix - h.loss_mix):>12s} {plus_minus(h.best0 - h.best1):>11s}  "
                       f"{h.arm0.mode().iloc[0]:g} / {switch_desc(h)}")
    return out


def bound_check(runs):
    out = ["", "Chapter 5 regret bounds"]
    rows = []
    for (gen, working, delta), g in runs.groupby(["gen", "working", "delta"]):
        for switches, oracle in enumerate(ORACLES):
            summary = bound_summary(g.loss_mix - g[oracle], g["T"], switches, delta)
            if summary is not None:
                rows.append((gen, working, delta, switches, *summary))
    t = pd.DataFrame(rows, columns=["gen", "working", "delta", "m", "max", "bound", "viol", "n"])
    for (delta, switches), g in t.groupby(["delta", "m"]):
        worst = g.loc[g["max"].idxmax()]
        ratio = g["max"] / g.bound
        out.append(f"delta={delta:g} m={switches}: bound {g.bound.max():.2f}; max over all combos {g['max'].max():.2f} "
                   f"({worst.gen} x {worst.working}); violations {g.viol.sum()}/{g.n.sum()}; "
                   f"max/bound by combo min {ratio.min():.2f} max {ratio.max():.2f}")
    return out


def numerics(runs):
    clamped = runs[runs.n_clamp > 0].groupby(["gen", "working", "scenario"]).n_clamp.agg(["count", "max"])
    return ["", f"Non-finite arm log densities (guard used)  {int(runs.n_nonfinite.sum())} over {len(runs)} run-rows",
            "scored ticks with an arm at its parameter clamp (runs, max ticks per run):", clamped.to_string() if len(clamped) else "none"]


def regret_reports():
    write_report(R + "regret_2026-11-14.txt", gaussian_regret(pd.read_csv(R + "regret_seeds.csv")))
    runs = pd.concat([pd.read_csv(R + f"regretfam_seeds{C.sfx(f)}.csv") for f in FAMILIES])
    write_report(R + "regretfam_2026-11-14.txt", family_table(runs) + bound_check(runs) + numerics(runs))


def main():
    scoresfam()
    if reports_enabled():
        regret_reports()


if __name__ == "__main__":
    main()
