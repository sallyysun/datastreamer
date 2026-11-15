"""Chapter 6.3.2 estimation accuracy of fixed forgetting, VFF and MFF, from results/strategies_seeds[_fam].csv
(probe_strategies.cpp: Section 6.2 construction, cumulative KLL k = 256 margins, 60 seeds, prequential forecasts).
Fixed = the hindsight-best of the five fixed memories, chosen per family x estimator x scenario x window x metric (an
accuracy reference, not an online procedure and not regret). Family averages are taken after the per-family results:
per seed, the mean over the four families of each strategy's metric; SEs from those per-seed means (the families'
streams are independent, so this is the SE of the family-averaged paired difference).
Usage (from experiments/): python3 analyze_strategies.py -> results/strategies_2026-11-12.txt, results/strategies_summary.csv"""
import math
import os
import numpy as np
import pandas as pd

R = "results/"
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""   # mix: MFF density = mixture (MIXDENS run)
FAMS = [("gaussian", ""), ("clayton", "_clayton"), ("gumbel", "_gumbel"), ("t", "_t")]
FL = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t4"}
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
FIXED = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
MET = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
       ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]
WINS = {"constant": ["whole"], "drift": ["whole"], "jump": ["whole", "pre", "rec_0_100", "rec_100_500", "post"]}

d = pd.concat([pd.read_csv(R + f"strategies_seeds{T}{s}.csv") for _, s in FAMS])
L = []; P = lambda s="": L.append(s)
rows = []


def se(x):
    return x.std(ddof=1) / math.sqrt(len(x))


def cell(f, sc, e, w, m, scale):
    p = d[(d.copula == f) & (d.scenario == sc) & (d.estimator == e) & (d.window == w)].pivot(index="seed", columns="strategy", values=m) * scale
    best = min(FIXED, key=lambda k: p[k].mean())
    return pd.DataFrame({"fixed": p[best], "vff_slow": p["vff_slow"], "vff_fast": p["vff_fast"], "mff": p["mff"]}), best, p


P("Chapter 6.3.2 estimation accuracy. 60 seeds (42-101), matched paths, forecasts held before each observation,")
P("RMPL and all MFF/VFF signals on cumulative DataSketches KLL k = 256 pseudo-observations; Linux/libstdc++.")
P("Excess log loss and Hellinger: fixed and VFF use the Gaussian copula at their forecast rho (RMPL's own density; a plug-in")
P("working model for the counting matrix and Hermite series). MFF: " + ("the mixture sum_k w_k c_G(rho_k) of its candidates' Gaussian" if T else "the Gaussian copula at its forecast rho."))
if T: P("copula densities (weights held before each observation). Fixed = hindsight-best fixed memory per cell and metric.")
else: P("Fixed = hindsight-best fixed memory per cell and metric.")
P("Differences are paired (same streams): mean ± 2 SE.")
for m, mlab, scale in MET:
    P(); P("#" * 100); P(f"# {mlab}"); P("#" * 100)
    for sc, wins in WINS.items():
        for w in wins:
            P(); P(f"=== {sc}, window {w} ===")
            P(f"{'family':9s} {'estimator':16s} {'Fixed (N)':>14s} {'VFF slow':>12s} {'VFF fast':>12s} {'MFF':>12s} {'MFF-Fixed':>14s} {'MFF-VFFslow':>14s}")
            fam_tables = {}
            for e, el in EST:
                per = []
                for f, _ in FAMS:
                    t, best, _ = cell(f, sc, e, w, m, scale)
                    per.append(t)
                    mu = t.mean(); s = t.apply(se)
                    d1 = t.mff - t.fixed; d2 = t.mff - t.vff_slow
                    P(f"{FL[f]:9s} {el:16s} {mu.fixed:7.2f}({best[6:]:>4s}) {mu.vff_slow:7.2f}({s.vff_slow:.2f}) {mu.vff_fast:7.2f}({s.vff_fast:.2f}) "
                      f"{mu.mff:7.2f}({s.mff:.2f}) {d1.mean():+7.2f}±{2*se(d1):.2f} {d2.mean():+7.2f}±{2*se(d2):.2f}")
                    for k in ["fixed", "vff_slow", "vff_fast", "mff"]:
                        rows.append(dict(metric=m, scenario=sc, window=w, estimator=e, copula=f, strategy=k,
                                         mean=mu[k], se=s[k], best_fixed=best))
                avg = sum(per) / len(per)
                fam_tables[e] = avg
                mu = avg.mean(); s = avg.apply(se); d1 = avg.mff - avg.fixed; d2 = avg.mff - avg.vff_slow; d3 = avg.mff - avg.vff_fast
                P(f"{'mean of 4':9s} {el:16s} {mu.fixed:7.2f}{'':6s} {mu.vff_slow:7.2f}({s.vff_slow:.2f}) {mu.vff_fast:7.2f}({s.vff_fast:.2f}) "
                  f"{mu.mff:7.2f}({s.mff:.2f}) {d1.mean():+7.2f}±{2*se(d1):.2f} {d2.mean():+7.2f}±{2*se(d2):.2f}   (MFF-VFFfast {d3.mean():+.2f}±{2*se(d3):.2f})")
                for k in ["fixed", "vff_slow", "vff_fast", "mff"]:
                    rows.append(dict(metric=m, scenario=sc, window=w, estimator=e, copula="mean4", strategy=k, mean=mu[k], se=s[k], best_fixed=""))
pd.DataFrame(rows).to_csv(R + f"strategies_summary{T}.csv", index=False)

# Sign agreement of MFF - Fixed across metrics (whole run), per family x estimator x scenario
P(); P("=== agreement across metrics: sign of MFF - hindsight Fixed at 2 SE (W = MFF lower), whole run ===")
for sc in WINS:
    out = []
    for e, el in EST:
        for f, _ in FAMS:
            v = []
            for m, _, scale in MET:
                t, _, _ = cell(f, sc, e, "whole", m, scale)
                dd = t.mff - t.fixed
                v.append("W" if dd.mean() < -2 * se(dd) else "L" if dd.mean() > 2 * se(dd) else "T")
            out.append("".join(v))
    P(f"{sc:9s} (RMSE/excess/H2 per family x estimator): " + " ".join(out))
open(R + ("strategies_2026-11-12.txt" if not T else f"strategies{T}_2026-11-14.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
