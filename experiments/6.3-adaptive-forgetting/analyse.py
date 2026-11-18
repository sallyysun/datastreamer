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
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""
FIXED = [f"fixed_{m}" for m in C.MEMORIES]
STRATS = ["fixed", "vff_slow", "vff_fast", "mff"]
RATES = [f"{r:g}" for r in C.VFF_RATES]
MET = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
       ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]
SCENARIO_WINDOWS = [(sc, w) for sc in C.SCENARIOS for w in C.WINDOWS[sc]]


def wtl(x):
    return win_loss_tie(x.mean(), se(x))


def per_seed(d, f, sc, e, w, m, scale):
    """Per-seed metric of every strategy in one cell, and the fixed memory with the lowest mean."""
    p = d[(d.copula == f) & (d.scenario == sc) & (d.estimator == e) & (d.window == w)]
    p = p.pivot(index="seed", columns="strategy", values=m) * scale
    return p, min(FIXED, key=lambda k: p[k].mean())


def strategy_cell(d, f, sc, e, w, m, scale):
    p, best = per_seed(d, f, sc, e, w, m, scale)
    return pd.DataFrame({"fixed": p[best], "vff_slow": p["vff_slow"], "vff_fast": p["vff_fast"], "mff": p["mff"]}), best


def strategy_header():
    mff = "the mixture of its candidates' Gaussian densities" if T else "the Gaussian at its forecast"
    return [f"6.3 accuracy, 60 seeds. Density: fixed and VFF use the Gaussian at their forecast, MFF {mff}.",
            "Fixed = hindsight-best memory per cell and metric. Differences paired, mean ± 2 SE."]


STRATEGY_COLUMNS = (f"{'family':9s} {'estimator':16s} {'Fixed (N)':>14s} {'VFF slow':>12s} "
                    f"{'VFF fast':>12s} {'MFF':>12s} {'MFF-Fixed':>14s} {'MFF-VFFslow':>14s}")


def pm(x):
    return f"{x.mean():+7.2f}±{2 * se(x):.2f}"


def strategy_line(label, e, t, best=None):
    """One table row; best is the hindsight memory, or None for the family mean (which adds MFF - VFF fast)."""
    mu, s = t.mean(), t.apply(se)
    fixed = f"{mu.fixed:7.2f}" + (f"({best[6:]:>4s})" if best else " " * 6)
    line = (f"{label:9s} {C.EST_LABEL[e]:16s} {fixed} {mu.vff_slow:7.2f}({s.vff_slow:.2f}) {mu.vff_fast:7.2f}({s.vff_fast:.2f}) "
            f"{mu.mff:7.2f}({s.mff:.2f}) {pm(t.mff - t.fixed)} {pm(t.mff - t.vff_slow)}")
    if best is None:
        d3 = t.mff - t.vff_fast
        line += f"   (MFF-VFFfast {d3.mean():+.2f}±{2 * se(d3):.2f})"
    return line


def summary_rows(key, t, best):
    mu, s = t.mean(), t.apply(se)
    return [key | dict(strategy=k, mean=mu[k], se=s[k], best_fixed=best) for k in STRATS]


def strategy_tables(d):
    out, rows = [], []
    for m, mlab, scale in MET:
        out += ["", mlab, STRATEGY_COLUMNS]
        for sc, w in SCENARIO_WINDOWS:
            out += ["", f"{sc} / {w}"]
            for e in C.ESTIMATORS:
                key = dict(metric=m, scenario=sc, window=w, estimator=e)
                cells = {f: strategy_cell(d, f, sc, e, w, m, scale) for f in C.FAMILIES}
                for f, (t, best) in cells.items():
                    out.append(strategy_line(C.FAMILY_TEXT[f], e, t, best))
                    rows += summary_rows(key | {"copula": f}, t, best)
                avg = sum(t for t, _ in cells.values()) / len(cells)   # per seed, the mean over the four families
                out.append(strategy_line("mean of 4", e, avg))
                rows += summary_rows(key | {"copula": "mean4"}, avg, "")
    return out, pd.DataFrame(rows)


def mff_vs_fixed(d, sc, e, f, m, scale):
    t, _ = strategy_cell(d, f, sc, e, "whole", m, scale)
    return wtl(t.mff - t.fixed)


def agreement(d):
    out = ["", "MFF vs hindsight fixed by metric (whole run, W = lower loss at 2 SE)"]
    for sc in C.WINDOWS:
        codes = ["".join(mff_vs_fixed(d, sc, e, f, m, scale) for m, _, scale in MET)
                 for e, f in itertools.product(C.ESTIMATORS, C.FAMILIES)]
        out.append(f"{sc:9s} (RMSE/excess/H2 per family x estimator): " + " ".join(codes))
    return out


def strategies():
    d = pd.concat([pd.read_csv(R + f"strategies_seeds{T}{C.sfx(f)}.csv") for f in C.FAMILIES])
    body, summary = strategy_tables(d)
    summary.to_csv(R + f"strategies_summary{T}.csv", index=False)
    if reports_enabled():
        lines = strategy_header() + body + agreement(d)
        write_report(R + ("strategies_2026-11-12.txt" if not T else f"strategies{T}_2026-11-14.txt"), lines)


def load_vffsens(kind):
    return pd.concat([pd.read_csv(R + f"strategies_{kind}_vffsens{T[1:]}{C.sfx(f)}.csv") for f in C.FAMILIES])


def rate_table(d, m, scale, sc, w, e, fam):
    """Per-seed VFF at each alpha, MFF and hindsight-best fixed; fam is a family or 'mean4' (per-seed family mean)."""
    def one(f):
        p, best = per_seed(d, f, sc, e, w, m, scale)
        out = pd.DataFrame({a: p[f"vff_{a}"] for a in RATES})
        out["mff"] = p["mff"]
        out["fixed"] = p[best]
        return out
    return sum(one(f) for f in C.FAMILIES) / 4 if fam == "mean4" else one(fam)


def rate_summary(d):
    rows = []
    cells = itertools.product(MET, SCENARIO_WINDOWS, C.ESTIMATORS, C.FAMILIES + ["mean4"])
    for (m, _, scale), (sc, w), e, fam in cells:
        t = rate_table(d, m, scale, sc, w, e, fam)
        key = dict(metric=m, scenario=sc, window=w, estimator=e, copula=fam)
        for a in RATES:
            dd = t.mff - t[a]
            rows.append(key | dict(strategy=f"vff_{a}", mean=t[a].mean(), se=se(t[a]),
                                   mff_minus=dd.mean(), mff_minus_se=se(dd), verdict=wtl(dd)))
        for k in ("mff", "fixed"):
            rows.append(key | dict(strategy=k, mean=t[k].mean(), se=se(t[k])))
    return pd.DataFrame(rows)


def vff_means(S, m, sc, e):
    g = S[(S.metric == m) & (S.scenario == sc) & (S.window == "whole") & (S.estimator == e) & (S.copula == "mean4")
          & S.strategy.str.startswith("vff_")]
    return g.set_index("strategy")["mean"]


def rate_lines(S, m, sc, w, e):
    """Family-mean VFF at each alpha, MFF and fixed; then MFF - VFF with its verdict and 2 SE."""
    g = S[(S.metric == m) & (S.scenario == sc) & (S.window == w) & (S.estimator == e) & (S.copula == "mean4")].set_index("strategy")
    vff = [g.loc["vff_" + a] for a in RATES]
    return [f"{C.EST_LABEL[e]:16s}   " + " ".join(f"{v['mean']:7.2f}" for v in vff)
            + f" | {g.loc['mff', 'mean']:7.2f} {g.loc['fixed', 'mean']:7.2f}",
            f"{'':16s}   " + " ".join(f"{v.mff_minus:+6.2f}{v.verdict}" for v in vff),
            f"{'':16s}   " + " ".join(f"±{2 * v.mff_minus_se:5.2f} " for v in vff)]


def sensitivity(S):
    out = ["VFF learning-rate sensitivity, 60 seeds, MFF eta = 1, delta = 1e-4. alpha = 0.02 is the development choice.",
           "MFF − VFF paired, mean ± 2 SE; W/T/L = MFF lower / within 2 SE / higher."]
    columns = f"{'Estimator':16s} a=" + " ".join(f"{a:>7s}" for a in RATES) + f" | {'MFF':>7s} {'Fixed':>7s}"
    for m, mlab, _ in MET:
        out += ["", mlab, columns]
        for sc, w in SCENARIO_WINDOWS:
            out += ["", f"{sc} / {w}"]
            for e in C.ESTIMATORS:
                out += rate_lines(S, m, sc, w, e)
    return out


def best_alpha(S):
    """Hindsight-best alpha per scenario, and the single alpha minimising the mean ratio to the per-scenario best."""
    out, rows = ["", "Hindsight-best alpha (whole run, family mean)"], []
    for m, mlab, _ in MET:
        out.append(f"-- {mlab}")
        for e in C.ESTIMATORS:
            means = {sc: vff_means(S, m, sc, e) for sc in C.SCENARIOS}
            cells = []
            for sc, g in means.items():
                a = g.idxmin()[4:]
                cells.append(f"{sc}: {a:>5s} ({g.min():.2f})")
                rows.append(dict(metric=m, estimator=e, scenario=sc, best_alpha=a, value=g.min()))
            ratios = {a: [g["vff_" + a] / g.min() for g in means.values()] for a in RATES}
            rel = {a: (np.mean(r), np.max(r)) for a, r in ratios.items()}
            a1 = min(rel, key=lambda a: rel[a][0])
            rows.append(dict(metric=m, estimator=e, scenario="all (single alpha)", best_alpha=a1, value=rel[a1][0]))
            out.append(f"   {C.EST_LABEL[e]:16s} " + "   ".join(cells) + f"   | best single alpha: {a1} (mean ratio to per-scenario best {rel[a1][0]:.3f}, "
                       f"worst {rel[a1][1]:.3f}); 0.05: {rel['0.05'][0]:.3f}/{rel['0.05'][1]:.3f}, 0.075: {rel['0.075'][0]:.3f}/{rel['0.075'][1]:.3f}, "
                       f"0.02: {rel['0.02'][0]:.3f}/{rel['0.02'][1]:.3f}")
    return out, pd.DataFrame(rows)


def comparisons(d, B):
    out = ["", "MFF - VFF (whole run, family mean ± 2 SE)"]
    for m, mlab, scale in MET:
        out.append(f"-- {mlab}")
        for e in C.ESTIMATORS:
            a1 = B[(B.metric == m) & (B.estimator == e) & (B.scenario == "all (single alpha)")].best_alpha.iloc[0]
            out.append(f"   {C.EST_LABEL[e]}:")
            for sc in C.SCENARIOS:
                ab = B[(B.metric == m) & (B.estimator == e) & (B.scenario == sc)].best_alpha.iloc[0]
                t = rate_table(d, m, scale, sc, "whole", e, "mean4")
                parts = []
                for lab, a in [("0.02(dev)", "0.02"), (f"best {ab}", ab), (f"single {a1}", a1)]:
                    dd = t["mff"] - t[a]
                    parts.append(f"{lab} {dd.mean():+.2f}±{2*se(dd):.2f}{wtl(dd)}")
                allw = "".join(wtl(t["mff"] - t[a]) for a in RATES)
                out.append("      " + f"{sc}: " + ", ".join(parts) + f", every alpha {allw}")
    return out


def robustness(d):
    out = ["", "MFF vs hindsight-best alpha per family (whole run)"]
    for m, mlab, scale in MET:
        v = {sc: [] for sc in C.SCENARIOS}
        for sc, e, f in itertools.product(C.SCENARIOS, C.ESTIMATORS, C.FAMILIES):
            t = rate_table(d, m, scale, sc, "whole", e, f)
            ab = min(RATES, key=lambda a: t[a].mean())
            v[sc].append(wtl(t["mff"] - t[ab]))
        counts = "  ".join(f"{sc} {c.count('W')}/{c.count('T')}/{c.count('L')}" for sc, c in v.items())
        out.append(f"{mlab} (W/T/L)  {counts}")
    return out


def extremes(X):
    out = ["", "Excess log loss extremes (untrimmed): seeds and seed-ticks with |rho_hat| > 0.98, summed over families;",
           "top-1% = share of the excess from the worst 1% of seed-ticks."]
    for e in C.ESTIMATORS:
        for sc in C.SCENARIOS:
            g = X[(X.scenario == sc) & (X.estimator == e)]
            out.append(f"-- {C.EST_LABEL[e]}, {sc}")
            for k in [f"vff_{a}" for a in RATES] + ["mff"]:
                h = g[g.strategy == k]
                out.append(f"   {k:10s} seeds {int(h.seeds_extreme.sum()):3d}  seed-ticks {int(h.seedticks_extreme.sum()):5d}  max {h.ex_max.max():9.0f}"
                           f"  mean {h.ex_mean.mean():8.2f}  median {h.ex_median.mean():6.2f}  top-1% share {h.top1pct_share.mean():5.2f}")
    return out


def vffsens():
    d = load_vffsens("seeds")
    S = rate_summary(d)
    S.to_csv(R + f"vffsens_summary{T}.csv", index=False)
    best, B = best_alpha(S)
    B.to_csv(R + f"vffsens_best_alpha{T}.csv", index=False)
    X = load_vffsens("extremes")
    X.to_csv(R + f"vffsens_extremes{T}.csv", index=False)
    if reports_enabled():
        lines = sensitivity(S) + best + comparisons(d, B) + robustness(d) + extremes(X)
        write_report(R + ("vffsens_2026-11-12.txt" if not T else f"vffsens{T}_2026-11-14.txt"), lines)


def main():
    strategies()
    vffsens()


if __name__ == "__main__":
    main()
