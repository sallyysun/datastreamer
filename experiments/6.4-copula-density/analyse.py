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
FAM = C.FAMILIES_T31
SC = C.SCENARIOS
GL = {"gaussian": "Gauss", "t": "t4", "clayton": "Clay", "gumbel": "Gumb"}
EL = {"rmpl": "RMPL (inv)", "xiao": "Counting Matrix (inv)", "hermite": "Hermite (inv)"}
MET = [("rmse", "RMSE x1e-2", 100), ("h2_x1000", "H2 x1e-3", 1), ("excess_x1000", "excess x1e-3", 1)]
VERDICTS = ("log", "tie", "sq")
WORKING_LABEL = {f: "RMPL " + C.FAMILY_TEXT[f] for f in FAM}
ORACLES = ("best0", "best1", "best2")
K = len(C.MEMORIES)


def load_scoresfam(tag):
    d = pd.concat([pd.read_csv(R + f"scoresfam_{tag}{C.sfx(f)}.csv") for f in FAM])
    return d[d.estimator.isin(C.ESTIMATORS)]


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
    for e in C.ESTIMATORS:
        out += [f"-- {EL[e]}", f"{'gen':6s} {'scenario':9s} " + " ".join(f"{ml:>17s}" for _, ml, _ in MET)]
        for g in FAM:
            for sc in SC:
                out.append(f"{GL[g]:6s} {sc:9s} " + " ".join(f"{fmt(cell(cells, g, sc, e, 'gaussian', m)):>17s}" for m, _, _ in MET))
    return out


def family_grid(cells, title, width, show):
    out = ["", title]
    for m, ml, _ in MET:
        for sc in SC:
            out.append(f"-- {ml}, {sc}")
            for e in C.ESTIMATORS:
                out += [f"   {EL[e]}", f"   {'gen \\ W':8s} " + " ".join(f"{GL[w]:>{width}s}" for w in FAM)]
                for g in FAM:
                    row = [show(cell(cells, g, sc, e, w, m), g == w) for w in FAM]
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
    for e in C.ESTIMATORS:
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
    for e, (m, ml, _) in itertools.product(C.ESTIMATORS, MET):
        cells = []
        for g, sc in itertools.product(FAM[1:], SC):
            a = d[(d.gen == g) & (d.scenario == sc) & (d.estimator == e) & (d.window == "whole") & (d.score == "log")]
            dd = (a[a.working == g].set_index("seed")[m] - a[a.working == "gaussian"].set_index("seed")[m]) * (100 if m == "rmse" else 1)
            estimate = format_estimate(dd.mean(), standard_error(dd, scale=2), "+.3f", ".3f", separator="±")
            cells.append(f"{GL[g]}/{sc[:5]} {estimate}")
        out.append(f"{EL[e]:22s} {ml:12s} " + "  ".join(cells))
    return out


def diagnostics(dg):
    cols = ["arm_clamp_ticks", "score_nonfinite", "eval_nonfinite"]
    return ["", "Numerical diagnostics (clamps from t = 500)",
            dg.groupby(["estimator", "working"])[cols].agg(["sum", "max"]).to_string()]


def eta_sensitivity(v):
    """Each score at its own dev-best eta, against the fixed eta = 1, on the dev seeds."""
    v = v[v.window == "whole"]
    out = ["", "Learning-rate sensitivity (dev seeds 1–30): best eta per score vs eta = 1"]
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


def scoresfam():
    d = load_scoresfam("seeds_sfeval")
    cells = paired_cells(d)
    cells.to_csv(R + "scoresfam_cells.csv", index=False)
    if not reports_enabled():
        return
    lines = ["MFF scoring rule by family. Δ = squared − log, paired over 60 seeds, mean ± 2 SE, * beyond 2 SE; positive = log better."]
    lines += gaussian_table(cells)
    lines += family_grid(cells, "Generating × working family (Δ, [ ] = matched)", 17,
                         lambda r, diag: f"[{fmt(r)}]" if diag else f" {fmt(r)} ")
    lines += family_grid(cells, "Mean accuracy (log / squared, * = matched)", 15,
                         lambda r, diag: f"{r['log']:6.3f}/{r['sq']:6.3f}" + ("*" if diag else " "))
    lines += verdict_counts(cells) + matching(d) + diagnostics(load_scoresfam("diag_sfeval"))
    if os.path.exists(R + "scoresfam_seeds_sfdev.csv"):
        lines += eta_sensitivity(load_scoresfam("seeds_sfdev"))
    write_report(R + "scoresfam_2026-11-15.txt", lines)


def bound(switches, delta, ticks):
    if delta == 0:
        return math.log(K) if switches == 0 else float("nan")
    return (math.log(K) + (ticks - 1 - switches) * math.log(1 / (1 - delta + delta / K))
            + switches * math.log(K / delta))


def regret(rows, oracle, forecast="mix"):
    return rows[f"loss_{forecast}"] - rows[oracle]


def bound_summary(loss, ticks, switches, delta):
    limits = np.array([bound(switches, delta, count) for count in ticks])
    if np.isnan(limits).all():
        return None
    loss = loss.to_numpy()
    return loss.max(), limits.max(), int((loss > limits + 1e-9).sum()), len(loss)


def rows_for_delta(rows, delta):
    return rows[np.isclose(rows.delta, delta)]


def paired_updates(rows):
    return rows_for_delta(rows, 1e-4).set_index("seed"), rows_for_delta(rows, 0.0).set_index("seed")


def switch_counts(runs):
    return runs.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False)


def gaussian_summary(x):
    estimate = format_estimate(x.mean(), 2 * standard_error(x), "7.2f", "5.2f")
    return f"{estimate} (med {x.median():6.2f})"


def regret_lines(g):
    out = []
    for d, dl in [(0.0, "Hedge      "), (1e-4, "fixed share")]:
        h = rows_for_delta(g, d)
        for m, oracle in enumerate(ORACLES):
            mix = regret(h, oracle)
            point = regret(h, oracle, "point")
            summary = bound_summary(mix, h["T"], m, d)
            violations = ""
            if summary is not None:
                _, limit, count, runs = summary
                violations = f" viol {count}/{runs} (bound {limit:.2f})"
            cells = [f"mix {gaussian_summary(mix)}{violations}", f"point {gaussian_summary(point)}"]
            out.append(f"   {dl} m={m}: " + " | ".join(cells))
    return out


def switch_line(g):
    j = g[g.delta == 0]
    sw = j[j.sw_t > 0]
    arms = ", ".join(f"{a:g}->{b:g} x{n}" for (a, b), n in switch_counts(sw).items())
    return (f"   best one-switch sequence: {len(sw)}/{len(j)} runs switch; switch time median {sw.sw_t.median():.0f} "
            f"(IQR {sw.sw_t.quantile(.25):.0f}-{sw.sw_t.quantile(.75):.0f}; true change 2500); arms: " + arms)


def gaussian_regret(r):
    out = ["MFF regret, Gaussian copula (nats, log loss, eta = 1). mean ± 2 SE (median); viol = runs above the Chapter 5 bound."]
    for e in C.ESTIMATORS:
        out += ["", f"{C.EST_LABEL[e]} (T = {sorted(r[r.estimator == e]['T'].unique())})"]
        for sc in SC:
            g = r[(r.estimator == e) & (r.scenario == sc)]
            gain = g[g.delta == 0].best0 - g[g.delta == 0].best1
            out.append(f"-- {sc}: best fixed arm (mode) {g.arm0.mode().iloc[0]:g}; best fixed - best one-switch loss {gaussian_summary(gain)}")
            out += regret_lines(g)
            a, b = paired_updates(g)
            out.append(f"   fixed share - Hedge (paired, = difference in cumulative MFF loss): mix {gaussian_summary(a.loss_mix - b.loss_mix)} | point {gaussian_summary(a.loss_point - b.loss_point)}")
            if sc == "jump":
                out.append(switch_line(g))
    return out


def family_summary(x):
    return format_estimate(x.mean(), 2 * standard_error(x), "6.2f", "4.2f", separator="±")


def switch_desc(h):
    sw = h[h.sw_t > 0]
    if not len(sw):
        return "no switch"
    top = switch_counts(sw).head(2)
    return f"{len(sw)}/60 switch, t~{sw.sw_t.median():.0f}, " + ",".join(f"{a:g}>{b:g}x{n}" for (a, b), n in top.items())


def family_table(r):
    out = ["MFF regret across families (nats), mean ± 2 SE. FS = fixed share, H = Hedge, sw1 = one-switch regret,",
           "gain1 = best fixed − best one-switch loss, [M] = matched."]
    for sc in SC:
        out += ["", sc,
                f"{'gen':9s} {'working':13s} {'stat FS':>11s} {'stat H':>11s} {'sw1 H':>11s} {'sw1 FS':>11s} {'FS - H':>12s} {'gain1':>11s}  best fixed / one-switch"]
        for gfam in FAM:
            for w in FAM:
                g = r[(r.gen == gfam) & (r.scenario == sc) & (r.working == w)]
                fs, h = paired_updates(g)
                wl = WORKING_LABEL[w] + (' [M]' if gfam == w else '')
                out.append(f"{C.FAMILY_TEXT[gfam]:9s} {wl:13s} {family_summary(regret(fs, 'best0')):>11s} {family_summary(regret(h, 'best0')):>11s} "
                           f"{family_summary(regret(h, 'best1')):>11s} {family_summary(regret(fs, 'best1')):>11s} {family_summary(fs.loss_mix - h.loss_mix):>12s} "
                           f"{family_summary(h.best0 - h.best1):>11s}  {h.arm0.mode().iloc[0]:g} / {switch_desc(h)}")
    return out


def bound_check(r):
    out = ["", "Chapter 5 regret bounds"]
    rows = []
    for (gfam, w, d), g in r.groupby(["gen", "working", "delta"]):
        for m, oracle in enumerate(ORACLES):
            summary = bound_summary(regret(g, oracle), g["T"], m, d)
            if summary is not None:
                rows.append((gfam, w, d, m, *summary))
    t = pd.DataFrame(rows, columns=["gen", "working", "delta", "m", "max", "bound", "viol", "n"])
    for (d, m), g in t.groupby(["delta", "m"]):
        i = g["max"].idxmax()
        out.append(f"delta={d:g} m={m}: bound {g.bound.max():.2f}; max over all combos {g['max'].max():.2f} ({g.loc[i, 'gen']} x {g.loc[i, 'working']}); "
                   f"violations {g.viol.sum()}/{g.n.sum()}; max/bound by combo min {(g['max']/g.bound).min():.2f} max {(g['max']/g.bound).max():.2f}")
    return out


def numerics(r):
    c = r[r.n_clamp > 0].groupby(["gen", "working", "scenario"]).n_clamp.agg(["count", "max"])
    return ["", f"Non-finite arm log densities (guard used)  {int(r.n_nonfinite.sum())} over {len(r)} run-rows",
            "scored ticks with an arm at its parameter clamp (runs, max ticks per run):", c.to_string() if len(c) else "none"]


def regret_reports():
    write_report(R + "regret_2026-11-14.txt", gaussian_regret(pd.read_csv(R + "regret_seeds.csv")))
    rows = pd.concat([pd.read_csv(R + f"regretfam_seeds{C.sfx(f)}.csv") for f in FAM])
    write_report(R + "regretfam_2026-11-14.txt", family_table(rows) + bound_check(rows) + numerics(rows))


def main():
    scoresfam()
    if reports_enabled():
        regret_reports()


if __name__ == "__main__":
    main()
