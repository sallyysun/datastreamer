"""Chapter 6.3 MFF scoring-function comparison (squared vs log prediction loss for the weight update), Gaussian copula,
KLL margins, fixed share delta = 1e-4, five arms N_eff 39-1999. results/scores_seeds_{dev,eval}.csv (probe_strategies
SCORES=1; dev = seeds 1-30, eval = seeds 42-101; log eta = 1 reproduces strategies_seeds MFF exactly).
eta selection (dev seeds only, the thesis_bank_tuning.md protocol): one eta per loss, shared by the three estimators,
minimising whole-run RMSE regret against the hindsight-best fixed arm averaged over drift and jump x 3 estimators.
Evaluation: squared eta = 1 vs log eta = 1 (main MFF; squared's dev optimum is also 1), plus log eta = 2 (log's dev
optimum) as sensitivity. Paired differences over the 60 eval seeds, mean ± 2 SE.
Usage (from experiments/): python3 analyze_scores.py -> results/scores_2026-11-14.txt"""
import math
import os
import numpy as np
import pandas as pd

R = "results/"
T = os.environ.get("DENS_TAG", "")   # mix: MFF density = mixture (scores_seeds_evalmix.csv); dev selection uses RMSE only
EL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
SC = ["constant", "drift", "jump"]
MET = [("rmse", "rho_S RMSE x1e-2", 100), ("h2_x1000", "H^2 x1e-3", 1), ("excess_x1000", "excess log loss x1e-3", 1)]
L = []; P = lambda s="": L.append(s)

dev = pd.read_csv(R + "scores_seeds_dev.csv"); dev = dev[dev.window == "whole"]
g = dev.groupby(["scenario", "estimator", "method", "eta"]).rmse.mean().reset_index()
best = g[g.method == "fixed"].groupby(["scenario", "estimator"]).rmse.min().rename("best")
m = g[g.method != "fixed"].join(best, on=["scenario", "estimator"])
m["regret"] = 1000 * (m.rmse - m.best)
sel = m[m.scenario.isin(["drift", "jump"])].groupby(["method", "eta"]).regret.mean()
P("MFF scoring function: squared vs log prediction loss. Gaussian copula, KLL margins, delta = 1e-4, arms 39-1999, Linux/libstdc++.")
P("MFF density forecast for H^2 / excess: " + ("mixture sum_k w_k c_G(rho_k)." if T else "Gaussian copula at the MFF rho (plug-in)."))
P(); P("=== eta selection, dev seeds 1-30: whole-run RMSE regret vs hindsight-best fixed arm (x1e-3), mean of drift + jump x 3 estimators ===")
for meth in ["squared", "log"]:
    s = sel[meth]; P(f"{meth:8s} " + "  ".join(f"eta {e:g}: {v:+.2f}" for e, v in s.items()) + f"   -> selected eta {s.idxmin():g}")

ev = pd.read_csv(R + f"scores_seeds_eval{T}.csv")
ARMS = {"sq": ("squared", 1.0), "log": ("log", 1.0), "log2": ("log", 2.0)}


def piv(sc, e, w, met, key):
    meth, eta = ARMS[key]
    r = ev[(ev.scenario == sc) & (ev.estimator == e) & (ev.window == w) & (ev.method == meth) & np.isclose(ev.eta, eta)]
    return r.set_index("seed")[met]


for e in ["rmpl", "xiao", "hermite"]:
    P(); P(f"=== {EL[e]}: eval seeds 42-101, mean (SE); paired diff = squared - log (eta 1), mean ± 2 SE; * beyond 2 SE ===")
    for met, ml, sc_ in MET:
        P(f"-- {ml}")
        P(f"{'scenario':9s} {'window':12s} {'squared':>14s} {'log':>14s} {'sq - log':>18s} {'log2':>14s} {'log2 - log':>18s}")
        for sc in SC:
            for w in (["whole"] if sc != "jump" else ["whole", "pre", "rec_0_100", "rec_100_500", "post"]):
                a, b, c = (piv(sc, e, w, met, k) * sc_ for k in ("sq", "log", "log2"))
                def f(x): return f"{x.mean():8.3f}({x.std(ddof=1)/math.sqrt(len(x)):.3f})"
                def dd(x):
                    mu, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x)); return f"{mu:+8.3f} ± {2*se:.3f}{'*' if abs(mu) > 2*se else ' '}"
                P(f"{sc:9s} {w:12s} {f(a):>14s} {f(b):>14s} {dd(a - b):>18s} {f(c):>14s} {dd(c - b):>18s}")
open(R + (f"scores_{T}_2026-11-14.txt" if T else "scores_2026-11-14.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
