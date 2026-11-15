"""Regret robustness over generating x working copula families (Chapter 6.3 appendix). results/regretfam_seeds[_fam].csv
(probe_strategies REGRETFAM=1, COPULA = generating family; seeds 42-101; matched paths; KLL margins). Working models,
five memories N_eff 39-1999 each: rmpl = the Chapter 6 Gaussian RMPL arms; cf_<family> = CopulaFit online MPL of that
family (expected Fisher, forgetting 1 - 2/(N+1), nu = 4; the family bank's arms; NOT RMPL, a different recursion; cf_gaussian
is the control for the recursion). Log loss, eta = 1, mixture forecaster; delta = 0 (Hedge) / 1e-4 (fixed share).
Regret over the scored ticks (t >= 30). Bound (Chapter 5): B_{T,m}(delta) = log K + (T-1-m) log 1/(1-delta+delta/K)
+ m log(K/delta); delta = 0: static log K only. The bound is pathwise (realised losses), so it needs no correct specification.
Usage (from experiments/): python3 analyze_regretfam.py -> results/regretfam_2026-11-14.txt"""
import math
import numpy as np
import pandas as pd

R = "results/"
r = pd.concat([pd.read_csv(R + f"regretfam_seeds{s}.csv") for s in ["", "_t", "_clayton", "_gumbel"]])
GEN = ["gaussian", "t", "clayton", "gumbel"]
WK = ["rmpl", "cf_gaussian", "cf_t", "cf_clayton", "cf_gumbel"]
MATCH = {"gaussian": "rmpl", "t": "cf_t", "clayton": "cf_clayton", "gumbel": "cf_gumbel"}
GL = {"gaussian": "Gaussian", "t": "t4", "clayton": "Clayton", "gumbel": "Gumbel"}
WL = {"rmpl": "RMPL (Gauss)", "cf_gaussian": "CF Gaussian", "cf_t": "CF t4", "cf_clayton": "CF Clayton", "cf_gumbel": "CF Gumbel"}
K = 5
L = []; P = lambda s="": L.append(s)


def B(m, d, T):
    if d == 0: return math.log(K) if m == 0 else np.nan
    return math.log(K) + (T - 1 - m) * math.log(1 / (1 - d + d / K)) + m * math.log(K / d)


def ms(x, med=False):
    s = f"{x.mean():6.2f}±{2 * x.std(ddof=1) / math.sqrt(len(x)):4.2f}"
    return s + (f" ({x.median():6.2f})" if med else "")


def piv(g, d): return g[np.isclose(g.delta, d)].set_index("seed")


P("Regret robustness (nats, cumulative over T = 4969-4970 scored ticks), mean ± 2 SE over 60 seeds, (median) where shown.")
P("FS = fixed share delta = 1e-4, H = Hedge delta = 0. stat FS = static regret of FS; sw1 = one-switch regret; FS-H = paired")
P("difference in cumulative mixture log loss; gain1 = best fixed - best one-switch comparator loss. [M] = correctly specified.")
for sc in ["constant", "drift", "jump"]:
    P(); P(f"=== {sc} ===")
    P(f"{'gen':9s} {'working':13s} {'stat FS':>11s} {'stat H':>11s} {'sw1 H':>11s} {'sw1 FS':>11s} {'FS - H':>12s} {'gain1':>11s}  best fixed / one-switch")
    for gfam in GEN:
        for w in WK:
            g = r[(r.gen == gfam) & (r.scenario == sc) & (r.working == w)]
            fs, h = piv(g, 1e-4), piv(g, 0.0)
            arm0 = h.arm0.mode().iloc[0]
            sw = h[h.sw_t > 0]
            swdesc = (f"{len(sw)}/60 switch, t~{sw.sw_t.median():.0f}, " + ",".join(f"{a:g}>{b:g}x{n}" for (a, b), n in
                      sw.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False).head(2).items())) if len(sw) else "no switch"
            P(f"{GL[gfam]:9s} {WL[w] + (' [M]' if MATCH[gfam] == w else ''):13s} {ms(fs.loss_mix - fs.best0):>11s} {ms(h.loss_mix - h.best0):>11s} "
              f"{ms(h.loss_mix - h.best1):>11s} {ms(fs.loss_mix - fs.best1):>11s} {ms(fs.loss_mix - h.loss_mix):>12s} {ms(h.best0 - h.best1):>11s}  {arm0:g} / {swdesc}")

P(); P("=== Chapter 5 bound check (every run): max regret / bound, violations ===")
rows = []
for (gfam, w, d), g in r.groupby(["gen", "working", "delta"]):
    for m, c in [(0, "best0"), (1, "best1"), (2, "best2")]:
        b = np.array([B(m, d, T) for T in g["T"]])
        if np.isnan(b).all(): continue
        x = (g.loss_mix - g[c]).values
        rows.append((gfam, w, d, m, x.max(), b.max(), int((x > b + 1e-9).sum()), len(x)))
t = pd.DataFrame(rows, columns=["gen", "working", "delta", "m", "max", "bound", "viol", "n"])
for (d, m), g in t.groupby(["delta", "m"]):
    i = g["max"].idxmax()
    P(f"delta={d:g} m={m}: bound {g.bound.max():.2f}; max over all combos {g['max'].max():.2f} ({g.loc[i, 'gen']} x {g.loc[i, 'working']}); "
      f"violations {g.viol.sum()}/{g.n.sum()}; max/bound by combo min {(g['max']/g.bound).min():.2f} max {(g['max']/g.bound).max():.2f}")
P(); P("=== numerics ===")
P(f"non-finite arm log densities (guard used): {int(r.n_nonfinite.sum())} over {len(r)} run-rows")
c = r[r.n_clamp > 0].groupby(["gen", "working", "scenario"]).n_clamp.agg(["count", "max"])
P("scored ticks with an arm at its parameter clamp (runs, max ticks per run):"); P(c.to_string() if len(c) else "none")
open(R + "regretfam_2026-11-14.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
