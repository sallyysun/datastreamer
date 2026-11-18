"""Compare squared and log loss across generating and working copulas."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import format_estimate, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402

R = "results/"
FS = [C.sfx(f) for f in C.FAMILIES_T31]
GEN = C.FAMILIES_T31
GL = {"gaussian": "Gauss", "t": "t4", "clayton": "Clay", "gumbel": "Gumb"}
EST = ["rmpl", "xiao", "hermite"]
EL = {"rmpl": "RMPL (inv)", "xiao": "Counting Matrix (inv)", "hermite": "Hermite (inv)"}
MET = [("rmse", "RMSE x1e-2", 100), ("h2_x1000", "H2 x1e-3", 1), ("excess_x1000", "excess x1e-3", 1)]
SC = C.SCENARIOS
VERDICTS = ("log", "tie", "sq")


def load(tag):
    d = pd.concat([pd.read_csv(R + f"scoresfam_{tag}{s}.csv") for s in FS])
    return d[d.estimator.isin(EST)]


def paired_cells(d):
    key = ["gen", "scenario", "estimator", "working", "window", "seed"]
    sq, lg = d[d.score == "squared"].set_index(key), d[d.score == "log"].set_index(key)
    rows = []
    for (g, sc, e, w, win), _ in sq.groupby(level=[0, 1, 2, 3, 4]):
        a, b = sq.loc[(g, sc, e, w, win)], lg.loc[(g, sc, e, w, win)]
        for m, _, s in MET:
            x, y = a[m] * s, b[m] * s
            dd = (x - y).dropna()
            mu, se = dd.mean(), standard_error(dd)
            rows.append(dict(gen=g, scenario=sc, estimator=e, working=w, window=win, metric=m, sq=x.mean(), log=y.mean(),
                             sq_med=x.median(), log_med=y.median(), diff=mu, se=se, diff_med=dd.median(),
                             verdict="log" if mu > 2 * se else "sq" if mu < -2 * se else "tie"))
    return pd.DataFrame(rows)


def fmt(r):
    estimate = format_estimate(r["diff"], 2 * r["se"], "+7.3f", ".3f", separator="±")
    return estimate + ("*" if r["verdict"] != "tie" else " ")


def tally_verdicts(rows):
    return rows.verdict.value_counts().reindex(VERDICTS, fill_value=0)


def counts(rows):
    return "/".join(map(str, tally_verdicts(rows)))


def cell(cells, g, sc, e, w, m):
    return cells[(cells.gen == g) & (cells.scenario == sc) & (cells.estimator == e) & (cells.working == w)
                 & (cells.window == "whole") & (cells.metric == m)].iloc[0]


def gaussian_table(cells):
    out = ["", "Gaussian working family by generating family (Δ)"]
    for e in EST:
        out += [f"-- {EL[e]}", f"{'gen':6s} {'scenario':9s} " + " ".join(f"{ml:>17s}" for _, ml, _ in MET)]
        for g in GEN:
            for sc in SC:
                out.append(f"{GL[g]:6s} {sc:9s} " + " ".join(f"{fmt(cell(cells, g, sc, e, 'gaussian', m)):>17s}" for m, _, _ in MET))
    return out


def family_grid(cells, title, width, show):
    out = ["", title]
    for m, ml, _ in MET:
        for sc in SC:
            out.append(f"-- {ml}, {sc}")
            for e in EST:
                out += [f"   {EL[e]}", f"   {'gen \\ W':8s} " + " ".join(f"{GL[w]:>{width}s}" for w in GEN)]
                for g in GEN:
                    row = [show(cell(cells, g, sc, e, w, m), g == w) for w in GEN]
                    out.append(f"   {GL[g]:8s} " + " ".join(f"{c:>{width}s}" for c in row))
    return out


def verdict_counts(cells):
    W0 = cells[cells.window == "whole"]
    out = ["", "Whole-run verdicts (log / tie / squared)"]
    for m, ml, _ in MET:
        rows = W0[W0.metric == m]
        log, tie, sq = tally_verdicts(rows)
        out.append(f"{ml:14s} log {log:3d}  tie {tie:3d}  sq {sq:3d}   (of {len(rows)})")
    out.append("by estimator:")
    for e in EST:
        out.append("  " + EL[e] + ": " + "; ".join(f"{ml} " + counts(W0[(W0.metric == m) & (W0.estimator == e)]) for m, ml, _ in MET))
    out.append("matched (diagonal) vs misspecified (off-diagonal):")
    for lab, mask in [("matched", W0.gen == W0.working), ("misspecified", W0.gen != W0.working)]:
        out.append("  " + lab + ": " + "; ".join(f"{ml} " + counts(W0[mask & (W0.metric == m)]) for m, ml, _ in MET))
    out.append("squared-loss wins (all windows):")
    sw = cells[cells.verdict == "sq"]
    cols = ["gen", "scenario", "estimator", "working", "window", "metric", "sq", "log", "diff", "se", "diff_med"]
    out.append(sw[cols].round(3).to_string(index=False) if len(sw) else "none")
    out += ["", "Jump-window verdicts (log / tie / squared)"]
    for win in ["pre", "rec_0_100", "rec_100_500", "post"]:
        x = cells[cells.window == win]
        out.append(f"{win:12s} " + "; ".join(f"{ml} " + counts(x[x.metric == m]) for m, ml, _ in MET))
    out += ["", "RMSE vs density verdict reversals"]
    pv = W0.pivot_table(index=["gen", "scenario", "estimator", "working"], columns="metric", values="verdict", aggfunc="first")
    rev = pv[(pv.rmse != pv.h2_x1000) | (pv.rmse != pv.excess_x1000)]
    return out + [f"{len(rev)} of {len(pv)} cells", rev.to_string() if len(rev) else ""]


def matching(d):
    out = ["", "Log-loss MFF (matched minus Gaussian working family)"]
    for e in EST:
        for m, ml, _ in MET:
            cells = []
            for g in GEN[1:]:
                for sc in SC:
                    a = d[(d.gen == g) & (d.scenario == sc) & (d.estimator == e) & (d.window == "whole") & (d.score == "log")]
                    dd = (a[a.working == g].set_index("seed")[m] - a[a.working == "gaussian"].set_index("seed")[m]) * (100 if m == "rmse" else 1)
                    estimate = format_estimate(dd.mean(), standard_error(dd, scale=2), "+.3f", ".3f", separator="±")
                    cells.append(f"{GL[g]}/{sc[:5]} {estimate}")
            out.append(f"{EL[e]:22s} {ml:12s} " + "  ".join(cells))
    return out


def diagnostics(dg):
    cols = ["arm_clamp_ticks", "score_nonfinite", "eval_nonfinite"]
    return ["", "Numerical diagnostics (clamps counted from t = 500)",
            dg.groupby(["estimator", "working"])[cols].agg(["sum", "max"]).to_string()]


def eta_sensitivity(v):
    """Each score at its own dev-best eta, against the fixed eta = 1, on the dev seeds."""
    v = v[v.window == "whole"]
    out = ["", "Learning-rate sensitivity (development seeds 1–30)",
           "Compares the best eta for each score, cell and metric with eta = 1."]
    mv = v.groupby(["gen", "scenario", "estimator", "working", "score", "eta"])[["rmse", "excess_x1000"]].mean().reset_index()
    rows = []
    for (g, sc, e, w), x in mv.groupby(["gen", "scenario", "estimator", "working"]):
        for m in ["rmse", "excess_x1000"]:
            bs, bl = x[x.score == "squared"].set_index("eta")[m], x[x.score == "log"].set_index("eta")[m]
            rows.append(dict(gen=g, scenario=sc, estimator=e, working=w, metric=m, eta_sq=bs.idxmin(), eta_log=bl.idxmin(),
                             d_tuned=bs.min() - bl.min(), d_eta1=bs.loc[1.0] - bl.loc[1.0]))
    O = pd.DataFrame(rows)
    O["rev"] = np.sign(O.d_tuned) != np.sign(O.d_eta1)
    for m in ["rmse", "excess_x1000"]:
        o = O[O.metric == m]
        out += [f"{m}: tuned favours log in {int((o.d_tuned > 0).sum())}/{len(o)} cells (eta = 1: {int((o.d_eta1 > 0).sum())}); sign reversals {int(o.rev.sum())}",
                "   dev-best eta, squared: " + str(o.eta_sq.value_counts().sort_index().to_dict()) + "   log: " + str(o.eta_log.value_counts().sort_index().to_dict())]
    rv = O[O.rev]
    return out + ([rv.round(4).to_string(index=False)] if len(rv) else [])


def main():
    d = load("seeds_sfeval")
    cells = paired_cells(d)
    cells.to_csv(R + "scoresfam_cells.csv", index=False)
    lines = ["MFF scoring function across copula families. Delta = squared - log (paired, 60 seeds, mean ± 2 SE; * beyond 2 SE;",
             "positive = log loss better). Whole run unless stated. Linux/libstdc++. inv = parameter inversion of rho_S into W."]
    lines += gaussian_table(cells)
    lines += family_grid(cells, "Generating × working family (Δ, brackets mark matched families)", 17,
                         lambda r, diag: f"[{fmt(r)}]" if diag else f" {fmt(r)} ")
    lines += family_grid(cells, "Mean accuracy (log / squared, * marks matched families)", 15,
                         lambda r, diag: f"{r['log']:6.3f}/{r['sq']:6.3f}" + ("*" if diag else " "))
    lines += verdict_counts(cells) + matching(d) + diagnostics(load("diag_sfeval"))
    if os.path.exists(R + "scoresfam_seeds_sfdev.csv"):
        lines += eta_sensitivity(load("seeds_sfdev"))
    write_report(R + "scoresfam_2026-11-15.txt", lines)


if __name__ == "__main__":
    main()
