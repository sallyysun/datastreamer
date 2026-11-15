"""Old (Gaussian plug-in at sum_k w_k rho_k) vs new (mixture sum_k w_k c_G(rho_k)) MFF density forecast: excess log loss
and H^2. results/strategies_seeds[_fam].csv vs strategies_seeds_mix[_fam].csv (probe_strategies MIXDENS=1 TAG=mix),
scores_seeds_eval.csv vs scores_seeds_evalmix.csv. rho_S RMSE, weights and every non-MFF number are identical (checked).
Fixed = hindsight-best fixed arm per metric, family, estimator, scenario (as strategies_summary). Paired over 60 seeds.
Usage (from experiments/): python3 compare_mixdens.py -> results/mixdens_2026-11-14.txt"""
import math
import numpy as np
import pandas as pd

R = "results/"
FAMS = [("gaussian", ""), ("t", "_t"), ("clayton", "_clayton"), ("gumbel", "_gumbel")]
EL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
MET = [("h2_x1000", "H2 x1e-3"), ("excess_x1000", "excess x1e-3")]
L = []; P = lambda s="": L.append(s)


def ms(x): return f"{x.mean():8.3f} ± {2 * x.std(ddof=1) / math.sqrt(len(x)):.3f}"


P("MFF density forecast: plug-in (old) vs mixture (new). mean ± 2 SE over 60 seeds; diff = new - old (paired).")
P("Verdict columns: sign of (MFF - hindsight-best fixed) and (MFF - VFF a=0.02), old -> new, at 2 SE (W = MFF better, L = worse, T = tie).")
flips = []
for fam, s in FAMS:
    old = pd.read_csv(R + f"strategies_seeds{s}.csv"); new = pd.read_csv(R + f"strategies_seeds_mix{s}.csv")
    for met, ml in MET:
        P(); P(f"=== {fam}: {ml} ===")
        for sc in ["constant", "drift", "jump"]:
            for w in (["whole"] if sc != "jump" else ["whole", "rec_0_100", "rec_100_500", "post"]):
                for e in ["rmpl", "xiao", "hermite"]:
                    def g(d, k): return d[(d.scenario == sc) & (d.window == w) & (d.estimator == e) & (d.strategy == k)].set_index("seed")[met]
                    mo, mn, v = g(old, "mff"), g(new, "mff"), g(old, "vff_slow")
                    fx = {k: g(old, k) for k in old.strategy.unique() if k.startswith("fixed_")}
                    wb = old[(old.scenario == sc) & (old.window == "whole") & (old.estimator == e) & old.strategy.str.startswith("fixed_")]
                    best = wb.groupby("strategy")[met].mean().idxmin(); f = fx[best]
                    def vd(x):
                        mu, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x)); return "W" if mu < -2 * se else "L" if mu > 2 * se else "T"
                    a, b, c, d2 = vd(mo - f), vd(mn - f), vd(mo - v), vd(mn - v)
                    if a != b or c != d2: flips.append(f"{fam} {met} {sc} {w} {EL[e]}: vs fixed {a}->{b}, vs VFF {c}->{d2}")
                    P(f"{sc:8s} {w:11s} {EL[e]:16s} old {mo.mean():8.3f}  new {mn.mean():8.3f}  diff {ms(mn - mo)} ({100*(mn.mean()/mo.mean()-1):+6.1f}%)"
                      f"  fixed[{best[6:]}] {f.mean():8.3f}  vs fixed {a}->{b}  vs VFF {c}->{d2}")
P(); P("=== verdict changes (MFF vs hindsight-best fixed / VFF) ===")
P("\n".join(flips) if flips else "none")

P(); P("=== scoring function (Gaussian, eta = 1): squared - log, old -> new density, mean ± 2 SE ===")
so, sn = pd.read_csv(R + "scores_seeds_eval.csv"), pd.read_csv(R + "scores_seeds_evalmix.csv")
for met, ml in MET:
    for e in ["rmpl", "xiao", "hermite"]:
        for sc in ["constant", "drift", "jump"]:
            for w in (["whole"] if sc != "jump" else ["whole", "rec_0_100", "post"]):
                def g(d, k): return d[(d.scenario == sc) & (d.window == w) & (d.estimator == e) & (d.method == k) & np.isclose(d.eta, 1)].set_index("seed")[met]
                P(f"{ml:12s} {EL[e]:16s} {sc:8s} {w:10s} old {ms(g(so,'squared') - g(so,'log'))}   new {ms(g(sn,'squared') - g(sn,'log'))}")
open(R + "mixdens_2026-11-14.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
