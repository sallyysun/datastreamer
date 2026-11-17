"""MFF scoring function (squared vs log loss) across generating x working copula families. results/scoresfam_seeds_sfeval
[_fam].csv (probe_strategies SCORESFAM=1 TAG=sfeval, COPULA = generating family; seeds 42-101; matched paths; KLL margins;
delta = 1e-4; eta = 1; MFF density = mixture of the candidates' working-family densities; excess at the true PIT; H^2 on the
81x81 grid). Estimators: rmpl / xiao (Counting Matrix) / hermite = parameter inversion of each arm's rho_S into the working
family W (estimator unchanged); cf = CopulaFit online MPL of family W (likelihood fitting, a different recursion).
Delta = squared - log, paired over seeds: positive favours log loss. Dev eta grid (seeds 1-30, no H^2):
results/scoresfam_seeds_sfdev[_fam].csv.
Usage: python3 analyze_scoresfam.py -> results/scoresfam_2026-11-15.txt, scoresfam_cells.csv"""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as cfg  # noqa: E402

R = "results/"
FS = [cfg.sfx(f) for f in cfg.FAMILIES_T31]
GEN = cfg.FAMILIES_T31
GL = {"gaussian": "Gauss", "t": "t4", "clayton": "Clay", "gumbel": "Gumb"}
EST = ["rmpl", "xiao", "hermite", "cf"]
EL = {"rmpl": "RMPL (inv)", "xiao": "Counting Matrix (inv)", "hermite": "Hermite (inv)", "cf": "CopulaFit (MPL)"}
MET = [("rmse", "RMSE x1e-2", 100), ("h2_x1000", "H2 x1e-3", 1), ("excess_x1000", "excess x1e-3", 1)]
SC = cfg.SCENARIOS
d = pd.concat([pd.read_csv(R + f"scoresfam_seeds_sfeval{s}.csv") for s in FS])
dg = pd.concat([pd.read_csv(R + f"scoresfam_diag_sfeval{s}.csv") for s in FS])
L = []; P = lambda s="": L.append(s)

# paired cells
key = ["gen", "scenario", "estimator", "working", "window", "seed"]
sq = d[d.score == "squared"].set_index(key); lg = d[d.score == "log"].set_index(key)
rows = []
for (g, sc, e, w, win), _ in sq.groupby(level=[0, 1, 2, 3, 4]):
    a, b = sq.loc[(g, sc, e, w, win)], lg.loc[(g, sc, e, w, win)]
    for m, _, s in MET:
        x, y = a[m] * s, b[m] * s
        dd = (x - y).dropna(); n = len(dd)
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(n)
        rows.append(dict(gen=g, scenario=sc, estimator=e, working=w, window=win, metric=m, sq=x.mean(), log=y.mean(),
                         sq_med=x.median(), log_med=y.median(), diff=mu, se=se, diff_med=dd.median(),
                         verdict="log" if mu > 2 * se else "sq" if mu < -2 * se else "tie"))
C = pd.DataFrame(rows)
C.to_csv(R + "scoresfam_cells.csv", index=False)


def fmt(r): return f"{r['diff']:+7.3f}±{2*r['se']:.3f}{'*' if r['verdict'] != 'tie' else ' '}"


P("MFF scoring function across copula families. Delta = squared - log (paired, 60 seeds, mean ± 2 SE; * beyond 2 SE;")
P("positive = log loss better). Whole run unless stated. Linux/libstdc++. inv = parameter inversion of rho_S into W; MPL = CopulaFit.")

P(); P("=== Table 1: Gaussian working family, by generating family (Delta, whole run) ===")
for e in EST:
    P(f"-- {EL[e]}")
    P(f"{'gen':6s} {'scenario':9s} " + " ".join(f"{ml:>17s}" for _, ml, _ in MET))
    for g in GEN:
        for sc in SC:
            c = C[(C.gen == g) & (C.scenario == sc) & (C.estimator == e) & (C.working == "gaussian") & (C.window == "whole")].set_index("metric")
            P(f"{GL[g]:6s} {sc:9s} " + " ".join(f"{fmt(c.loc[m]):>17s}" for m, _, _ in MET))

P(); P("=== Table 2: generating (rows) x working (cols), Delta, whole run; [] = matched family ===")
for m, ml, _ in MET:
    for sc in SC:
        P(f"-- {ml}, {sc}")
        for e in EST:
            P(f"   {EL[e]}")
            P(f"   {'gen \\ W':8s} " + " ".join(f"{GL[w]:>17s}" for w in GEN))
            for g in GEN:
                cells = []
                for w in GEN:
                    r = C[(C.gen == g) & (C.scenario == sc) & (C.estimator == e) & (C.working == w) & (C.window == "whole") & (C.metric == m)].iloc[0]
                    s = fmt(r); cells.append(f"[{s}]" if g == w else f" {s} ")
                P(f"   {GL[g]:8s} " + " ".join(f"{c:>17s}" for c in cells))

P(); P("=== Table 3: absolute accuracy, log / squared (means), whole run ===")
for m, ml, _ in MET:
    for sc in SC:
        P(f"-- {ml}, {sc}")
        for e in EST:
            P(f"   {EL[e]}")
            P(f"   {'gen \\ W':8s} " + " ".join(f"{GL[w]:>15s}" for w in GEN))
            for g in GEN:
                cells = []
                for w in GEN:
                    r = C[(C.gen == g) & (C.scenario == sc) & (C.estimator == e) & (C.working == w) & (C.window == "whole") & (C.metric == m)].iloc[0]
                    cells.append(f"{r['log']:6.3f}/{r['sq']:6.3f}" + ("*" if g == w else " "))
                P(f"   {GL[g]:8s} " + " ".join(f"{c:>15s}" for c in cells))

P(); P("=== Verdict counts (whole run): log wins / ties / squared wins, by metric ===")
W0 = C[C.window == "whole"]
for m, ml, _ in MET:
    v = W0[W0.metric == m].verdict.value_counts()
    P(f"{ml:14s} log {v.get('log', 0):3d}  tie {v.get('tie', 0):3d}  sq {v.get('sq', 0):3d}   (of {len(W0[W0.metric == m])})")
P("by estimator:")
for e in EST:
    P("  " + EL[e] + ": " + "; ".join(f"{ml} " + "/".join(str(W0[(W0.metric == m) & (W0.estimator == e)].verdict.value_counts().get(k, 0)) for k in ("log", "tie", "sq")) for m, ml, _ in MET))
P("matched (diagonal) vs misspecified (off-diagonal):")
for lab, mask in [("matched", W0.gen == W0.working), ("misspecified", W0.gen != W0.working)]:
    P("  " + lab + ": " + "; ".join(f"{ml} " + "/".join(str(W0[mask & (W0.metric == m)].verdict.value_counts().get(k, 0)) for k in ("log", "tie", "sq")) for m, ml, _ in MET))
P("squared-loss wins (all windows):")
sw = C[C.verdict == "sq"]
P(sw[["gen", "scenario", "estimator", "working", "window", "metric", "sq", "log", "diff", "se", "diff_med"]].round(3).to_string(index=False) if len(sw) else "none")

P(); P("=== Jump recovery windows: verdict counts (log/tie/sq) ===")
for win in ["pre", "rec_0_100", "rec_100_500", "post"]:
    x = C[(C.window == win)]
    P(f"{win:12s} " + "; ".join(f"{ml} " + "/".join(str(x[x.metric == m].verdict.value_counts().get(k, 0)) for k in ("log", "tie", "sq")) for m, ml, _ in MET))

P(); P("=== RMSE vs density reversals (whole run): cells where the RMSE verdict differs from the H2 or excess verdict ===")
pv = W0.pivot_table(index=["gen", "scenario", "estimator", "working"], columns="metric", values="verdict", aggfunc="first")
rev = pv[(pv.rmse != pv.h2_x1000) | (pv.rmse != pv.excess_x1000)]
P(f"{len(rev)} of {len(pv)} cells"); P(rev.to_string() if len(rev) else "")

P(); P("=== Does matching the family help? absolute log-loss MFF, matched minus Gaussian working family, whole run ===")
for e in EST:
    for m, ml, _ in MET:
        cells = []
        for g in GEN:
            for sc in SC:
                if g == "gaussian": continue
                a = d[(d.gen == g) & (d.scenario == sc) & (d.estimator == e) & (d.window == "whole") & (d.score == "log")]
                x = a[a.working == g].set_index("seed")[m]; y = a[a.working == "gaussian"].set_index("seed")[m]
                dd = (x - y) * (100 if m == "rmse" else 1)
                cells.append(f"{GL[g]}/{sc[:5]} {dd.mean():+.3f}±{2*dd.std(ddof=1)/math.sqrt(len(dd)):.3f}")
        P(f"{EL[e]:22s} {ml:12s} " + "  ".join(cells))

P(); P("=== Diagnostics (scored window t >= 500 for clamps) ===")
g = dg.groupby(["estimator", "working"])[["arm_clamp_ticks", "score_nonfinite", "eval_nonfinite", "cf_param_clamp_ticks"]].agg(["sum", "max"])
P(g.to_string())

if os.path.exists(R + "scoresfam_seeds_sfdev.csv"):
    v = pd.concat([pd.read_csv(R + f"scoresfam_seeds_sfdev{s}.csv") for s in FS])
    v = v[v.window == "whole"]
    P(); P("=== Learning-rate sensitivity (dev seeds 1-30; whole run; no H2). Each score at its own dev-best eta per cell and metric ===")
    P("Sign of squared(best eta) - log(best eta) vs the fixed eta = 1 sign on the same dev seeds; reversals listed.")
    mv = v.groupby(["gen", "scenario", "estimator", "working", "score", "eta"])[["rmse", "excess_x1000"]].mean().reset_index()
    out = []
    for (g, sc, e, w), x in mv.groupby(["gen", "scenario", "estimator", "working"]):
        for m in ["rmse", "excess_x1000"]:
            bs = x[x.score == "squared"].set_index("eta")[m]; bl = x[x.score == "log"].set_index("eta")[m]
            out.append(dict(gen=g, scenario=sc, estimator=e, working=w, metric=m, eta_sq=bs.idxmin(), eta_log=bl.idxmin(),
                            d_tuned=bs.min() - bl.min(), d_eta1=bs.loc[1.0] - bl.loc[1.0]))
    O = pd.DataFrame(out)
    O["rev"] = np.sign(O.d_tuned) != np.sign(O.d_eta1)
    for m in ["rmse", "excess_x1000"]:
        o = O[O.metric == m]
        P(f"{m}: tuned favours log in {int((o.d_tuned > 0).sum())}/{len(o)} cells (eta = 1: {int((o.d_eta1 > 0).sum())}); sign reversals {int(o.rev.sum())}")
        P("   dev-best eta, squared: " + str(o.eta_sq.value_counts().sort_index().to_dict()) + "   log: " + str(o.eta_log.value_counts().sort_index().to_dict()))
    rv = O[O.rev]
    if len(rv): P(rv.round(4).to_string(index=False))
open(R + "scoresfam_2026-11-15.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
