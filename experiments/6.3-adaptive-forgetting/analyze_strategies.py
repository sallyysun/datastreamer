"""Section 6.3 accuracy of fixed forgetting (hindsight-best memory per cell), VFF and MFF: RMSE, excess log loss and
squared Hellinger per family, estimator, scenario and window, with paired differences. Writes
results/strategies_2026-11-12.txt and results/strategies_summary.csv (DENS_TAG=mix: the mixture-density run)."""
import os
import sys
import pandas as pd

sys.path.insert(0, "../common")
from report_output import write_report  # noqa: E402
from summary_stats import standard_error as se  # noqa: E402
from summary_stats import win_loss_tie  # noqa: E402
import config as C  # noqa: E402

R = "results/"
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""
FAMS = [(f, C.sfx(f)) for f in C.FAMILIES]
FIXED = [f"fixed_{m}" for m in C.MEMORIES]
STRATS = ["fixed", "vff_slow", "vff_fast", "mff"]
MET = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
       ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]


def cell(d, f, sc, e, w, m, scale):
    """Per-seed metric of each strategy; 'fixed' is the fixed memory with the lowest mean."""
    p = d[(d.copula == f) & (d.scenario == sc) & (d.estimator == e) & (d.window == w)]
    p = p.pivot(index="seed", columns="strategy", values=m) * scale
    best = min(FIXED, key=lambda k: p[k].mean())
    return pd.DataFrame({"fixed": p[best], "vff_slow": p["vff_slow"], "vff_fast": p["vff_fast"], "mff": p["mff"]}), best


def header():
    mff = ("the mixture sum_k w_k c_G(rho_k) of its candidates' Gaussian" if T else "the Gaussian copula at its forecast rho.")
    out = ["Chapter 6.3.2 estimation accuracy. 60 seeds (42-101), matched paths, forecasts held before each observation,",
           "RMPL and all MFF/VFF signals on cumulative DataSketches KLL k = 256 pseudo-observations; Linux/libstdc++.",
           "Excess log loss and Hellinger: fixed and VFF use the Gaussian copula at their forecast rho (RMPL's own density; a plug-in",
           "working model for the counting matrix and Hermite series). MFF: " + mff]
    if T:
        out.append("copula densities (weights held before each observation). Fixed = hindsight-best fixed memory per cell and metric.")
    else:
        out.append("Fixed = hindsight-best fixed memory per cell and metric.")
    return out + ["Differences are paired (same streams): mean ± 2 SE."]


def tables(d):
    out, rows = [], []
    columns = (f"{'family':9s} {'estimator':16s} {'Fixed (N)':>14s} {'VFF slow':>12s} "
               f"{'VFF fast':>12s} {'MFF':>12s} {'MFF-Fixed':>14s} {'MFF-VFFslow':>14s}")
    for m, mlab, scale in MET:
        out += ["", mlab, columns]
        for sc, wins in C.WINDOWS.items():
            for w in wins:
                out += ["", f"{sc} / {w}"]
                for e in C.ESTIMATORS:
                    el, per = C.EST_LABEL[e], []
                    for f, _ in FAMS:
                        t, best = cell(d, f, sc, e, w, m, scale)
                        per.append(t)
                        mu, s = t.mean(), t.apply(se)
                        d1, d2 = t.mff - t.fixed, t.mff - t.vff_slow
                        out.append(f"{C.FAMILY_TEXT[f]:9s} {el:16s} {mu.fixed:7.2f}({best[6:]:>4s}) {mu.vff_slow:7.2f}({s.vff_slow:.2f}) "
                                   f"{mu.vff_fast:7.2f}({s.vff_fast:.2f}) {mu.mff:7.2f}({s.mff:.2f}) "
                                   f"{d1.mean():+7.2f}±{2*se(d1):.2f} {d2.mean():+7.2f}±{2*se(d2):.2f}")
                        rows += [dict(metric=m, scenario=sc, window=w, estimator=e, copula=f, strategy=k,
                                      mean=mu[k], se=s[k], best_fixed=best) for k in STRATS]
                    avg = sum(per) / len(per)   # per seed, the mean over the four (independent) families
                    mu, s = avg.mean(), avg.apply(se)
                    d1, d2, d3 = avg.mff - avg.fixed, avg.mff - avg.vff_slow, avg.mff - avg.vff_fast
                    out.append(f"{'mean of 4':9s} {el:16s} {mu.fixed:7.2f}{'':6s} {mu.vff_slow:7.2f}({s.vff_slow:.2f}) "
                               f"{mu.vff_fast:7.2f}({s.vff_fast:.2f}) {mu.mff:7.2f}({s.mff:.2f}) "
                               f"{d1.mean():+7.2f}±{2*se(d1):.2f} {d2.mean():+7.2f}±{2*se(d2):.2f}   "
                               f"(MFF-VFFfast {d3.mean():+.2f}±{2*se(d3):.2f})")
                    rows += [dict(metric=m, scenario=sc, window=w, estimator=e, copula="mean4", strategy=k,
                                  mean=mu[k], se=s[k], best_fixed="") for k in STRATS]
    return out, pd.DataFrame(rows)


def agreement(d):
    out = ["", "MFF vs hindsight fixed by metric (whole run, W = lower loss at 2 SE)"]
    for sc in C.WINDOWS:
        codes = []
        for e in C.ESTIMATORS:
            for f, _ in FAMS:
                v = ""
                for m, _, scale in MET:
                    t, _ = cell(d, f, sc, e, "whole", m, scale)
                    dd = t.mff - t.fixed
                    v += win_loss_tie(dd.mean(), se(dd))
                codes.append(v)
        out.append(f"{sc:9s} (RMSE/excess/H2 per family x estimator): " + " ".join(codes))
    return out


def main():
    d = pd.concat([pd.read_csv(R + f"strategies_seeds{T}{s}.csv") for _, s in FAMS])
    body, summary = tables(d)
    summary.to_csv(R + f"strategies_summary{T}.csv", index=False)
    lines = header() + body + agreement(d)
    write_report(R + ("strategies_2026-11-12.txt" if not T else f"strategies{T}_2026-11-14.txt"), lines)


if __name__ == "__main__":
    main()
