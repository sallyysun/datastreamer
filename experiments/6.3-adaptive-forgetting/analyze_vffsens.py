"""Summarize sensitivity to the VFF learning rate."""
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
FAMS = [f for f in C.FAMILIES]
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
RATES = [f"{r:g}" for r in C.VFF_RATES]
FIXED = [f"fixed_{m}" for m in C.MEMORIES]
MET = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
       ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]


def wtl(x):
    return win_loss_tie(x.mean(), se(x))


def load(kind):
    return pd.concat([pd.read_csv(R + f"strategies_{kind}_vffsens{T[1:]}{C.sfx(f)}.csv") for f in FAMS])


def table(d, m, scale, sc, w, e, fam):
    """Per-seed VFF at each alpha, MFF and hindsight-best fixed; fam is a family or 'mean4' (per-seed family mean)."""
    def one(f):
        p = d[(d.copula == f) & (d.scenario == sc) & (d.window == w) & (d.estimator == e)]
        p = p.pivot(index="seed", columns="strategy", values=m) * scale
        best = min(FIXED, key=lambda k: p[k].mean())
        out = pd.DataFrame({a: p[f"vff_{a}"] for a in RATES})
        out["mff"] = p["mff"]
        out["fixed"] = p[best]
        return out
    return sum(one(f) for f in FAMS) / 4 if fam == "mean4" else one(fam)


def summary(d):
    rows = []
    for m, _, scale in MET:
        for sc in C.SCENARIOS:
            for w in C.WINDOWS[sc]:
                for e, _ in EST:
                    for fam in FAMS + ["mean4"]:
                        t = table(d, m, scale, sc, w, e, fam)
                        for k in RATES + ["mff", "fixed"]:
                            r = dict(metric=m, scenario=sc, window=w, estimator=e, copula=fam,
                                     strategy=k if k in ("mff", "fixed") else f"vff_{k}", mean=t[k].mean(), se=se(t[k]))
                            if k in RATES:
                                dd = t["mff"] - t[k]
                                r.update(mff_minus=dd.mean(), mff_minus_se=se(dd), verdict=wtl(dd))
                            rows.append(r)
    return pd.DataFrame(rows)


def vff_means(S, m, sc, e):
    g = S[(S.metric == m) & (S.scenario == sc) & (S.window == "whole") & (S.estimator == e) & (S.copula == "mean4")
          & S.strategy.str.startswith("vff_")]
    return g.set_index("strategy")["mean"]


def sensitivity(S):
    out = ["VFF learning-rate sensitivity. 60 seeds (42-101), matched paths, prequential forecasts, KLL k = 256 margins (shared by",
           "RMPL arms, VFF controller and MFF scoring), MFF eta = 1, delta = 1e-4 (log loss) fixed throughout. Linux/libstdc++.",
           "alpha = 0.02 = development choice. 'best alpha' = HINDSIGHT on the evaluation seeds. MFF - VFF: paired, mean ± 2 SE;",
           "W/T/L = MFF lower / within 2 SE / higher. Reproduction: alpha 0.02, 0.1, MFF and fixed rows equal strategies_seeds exactly.",
           "Window tables show family means followed by MFF - VFF and its uncertainty."]
    columns = f"{'Estimator':16s} a=" + " ".join(f"{a:>7s}" for a in RATES) + f" | {'MFF':>7s} {'Fixed':>7s}"
    for m, mlab, _ in MET:
        out += ["", mlab, columns]
        for sc in C.SCENARIOS:
            for w in C.WINDOWS[sc]:
                out += ["", f"{sc} / {w}"]
                for e, el in EST:
                    g = S[(S.metric == m) & (S.scenario == sc) & (S.window == w) & (S.estimator == e) & (S.copula == "mean4")]
                    g = g.set_index("strategy")
                    vals = " ".join(f"{g.loc['vff_' + a, 'mean']:7.2f}" for a in RATES)
                    out += [f"{el:16s}   {vals} | {g.loc['mff', 'mean']:7.2f} {g.loc['fixed', 'mean']:7.2f}",
                            f"{'':16s}   " + " ".join(f"{g.loc['vff_' + a, 'mff_minus']:+6.2f}{g.loc['vff_' + a, 'verdict']}" for a in RATES),
                            f"{'':16s}   " + " ".join(f"±{2 * g.loc['vff_' + a, 'mff_minus_se']:5.2f} " for a in RATES)]
    return out


def best_alpha(S):
    """Hindsight-best alpha per scenario, and the single alpha minimising the mean ratio to the per-scenario best."""
    out, rows = ["", "Hindsight-best alpha (whole run, family mean)"], []
    for m, mlab, _ in MET:
        out.append(f"-- {mlab}")
        for e, el in EST:
            cells = []
            for sc in C.SCENARIOS:
                g = vff_means(S, m, sc, e)
                a = g.idxmin()[4:]
                cells.append(f"{sc}: {a:>5s} ({g.min():.2f})")
                rows.append(dict(metric=m, estimator=e, scenario=sc, best_alpha=a, value=g.min()))
            rel = {}
            for a in RATES:
                rr = []
                for sc in C.SCENARIOS:
                    g = vff_means(S, m, sc, e)
                    rr.append(g["vff_" + a] / g.min())
                rel[a] = (np.mean(rr), np.max(rr))
            a1 = min(rel, key=lambda a: rel[a][0])
            rows.append(dict(metric=m, estimator=e, scenario="all (single alpha)", best_alpha=a1, value=rel[a1][0]))
            out.append(f"   {el:16s} " + "   ".join(cells) + f"   | best single alpha: {a1} (mean ratio to per-scenario best {rel[a1][0]:.3f}, "
                       f"worst {rel[a1][1]:.3f}); 0.05: {rel['0.05'][0]:.3f}/{rel['0.05'][1]:.3f}, 0.075: {rel['0.075'][0]:.3f}/{rel['0.075'][1]:.3f}, "
                       f"0.02: {rel['0.02'][0]:.3f}/{rel['0.02'][1]:.3f}")
    return out, pd.DataFrame(rows)


def comparisons(d, B):
    out = ["", "MFF - VFF (whole run, family mean ± 2 SE)"]
    for m, mlab, scale in MET:
        out.append(f"-- {mlab}")
        for e, el in EST:
            a1 = B[(B.metric == m) & (B.estimator == e) & (B.scenario == "all (single alpha)")].best_alpha.iloc[0]
            out.append(f"   {el}:")
            for sc in C.SCENARIOS:
                ab = B[(B.metric == m) & (B.estimator == e) & (B.scenario == sc)].best_alpha.iloc[0]
                t = table(d, m, scale, sc, "whole", e, "mean4")
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
        for sc in C.SCENARIOS:
            for e, _ in EST:
                for f in FAMS:
                    t = table(d, m, scale, sc, "whole", e, f)
                    ab = min(RATES, key=lambda a: t[a].mean())
                    v[sc].append(wtl(t["mff"] - t[ab]))
        counts = "  ".join(f"{sc} {c.count('W')}/{c.count('T')}/{c.count('L')}" for sc, c in v.items())
        out.append(f"{mlab} (W/T/L)  {counts}")
    return out


def extremes(X):
    out = ["", "Excess log loss extremes (60 seeds × 4500 ticks per family, untrimmed)",
           "seeds / seed-ticks with |rho_hat| > 0.98 summed over families; max per-tick loss; mean and median of per-tick excess",
           "(averaged over families); share of the summed excess from the worst 1% of seed-ticks (averaged over families)."]
    for e, el in EST:
        for sc in C.SCENARIOS:
            g = X[(X.scenario == sc) & (X.estimator == e)]
            out.append(f"-- {el}, {sc}")
            for k in [f"vff_{a}" for a in RATES] + ["mff"]:
                h = g[g.strategy == k]
                out.append(f"   {k:10s} seeds {int(h.seeds_extreme.sum()):3d}  seed-ticks {int(h.seedticks_extreme.sum()):5d}  max {h.ex_max.max():9.0f}"
                           f"  mean {h.ex_mean.mean():8.2f}  median {h.ex_median.mean():6.2f}  top-1% share {h.top1pct_share.mean():5.2f}")
    return out


def main():
    d = load("seeds")
    S = summary(d)
    S.to_csv(R + f"vffsens_summary{T}.csv", index=False)
    best, B = best_alpha(S)
    B.to_csv(R + f"vffsens_best_alpha{T}.csv", index=False)
    X = load("extremes")
    X.to_csv(R + f"vffsens_extremes{T}.csv", index=False)
    if reports_enabled():
        lines = sensitivity(S) + best + comparisons(d, B) + robustness(d) + extremes(X)
        write_report(R + ("vffsens_2026-11-12.txt" if not T else f"vffsens{T}_2026-11-14.txt"), lines)


if __name__ == "__main__":
    main()
