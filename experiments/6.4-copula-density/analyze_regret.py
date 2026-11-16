"""Chapter 6.3 MFF regret (static and switching), log loss, eta = 1, delta = 0 (ordinary Hedge) and 1e-4 (fixed share),
Gaussian copula, KLL margins, arms N_eff 39-1999, seeds 42-101. results/regret_{seeds,paths}.csv (probe_strategies
REGRET=1). Regret = cumulative MFF loss - comparator loss over the scored ticks (t0 = 30, or 100 for the counting matrix,
whose cut-points calibrate first), comparator = best arm (m = 0) or best arm sequence with <= m switches (exact DP), on the
same guarded arm losses. Two MFF losses: mix = -log sum_k w_k c_k (the Chapter 5 forecaster) and point =
-log c(sum_k w_k rho_k) (the Chapter 6 MFF's forecast). Bound (Chapter 5): B_{T,m}(delta) = log K + (T-1-m) log 1/(1-delta
+delta/K) + m log(K/delta), eta = 1; with delta = 0 only the static bound log K.
Usage: python3 analyze_regret.py -> results/regret_2026-11-14.txt"""
import math
import numpy as np
import pandas as pd

R = "results/"
EL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
r = pd.read_csv(R + "regret_seeds.csv")
K = 5
L = []; P = lambda s="": L.append(s)


def B(m, d, T):
    if d == 0: return math.log(K) if m == 0 else float("nan")
    return math.log(K) + (T - 1 - m) * math.log(1 / (1 - d + d / K)) + m * math.log(K / d)


def ms(x):
    return f"{x.mean():7.2f} ± {2 * x.std(ddof=1) / math.sqrt(len(x)):5.2f} (med {x.median():6.2f})"


P("MFF regret (nats, cumulative over the scored ticks). Log loss, eta = 1. Gaussian copula, KLL margins, 60 seeds, Linux/libstdc++.")
P("mean ± 2 SE (median). 'viol' = runs above the Chapter 5 bound (only defined for the mixture forecaster; delta = 0 has no switching bound).")
for e in ["rmpl", "xiao", "hermite"]:
    P(); P(f"=== {EL[e]} (T = {sorted(r[r.estimator == e]['T'].unique())}) ===")
    for sc in ["constant", "drift", "jump"]:
        g0 = r[(r.estimator == e) & (r.scenario == sc)]
        gain = g0[g0.delta == 0].best0 - g0[g0.delta == 0].best1
        P(f"-- {sc}: best fixed arm (mode) {g0.arm0.mode().iloc[0]:g}; best fixed - best one-switch loss {ms(gain)}")
        for d, dl in [(0.0, "Hedge      "), (1e-4, "fixed share")]:
            g = g0[np.isclose(g0.delta, d)]
            for m, c in [(0, "best0"), (1, "best1"), (2, "best2")]:
                cells = []
                for loss in ["mix", "point"]:
                    x = g[f"loss_{loss}"] - g[c]
                    bnd = np.array([B(m, d, T) for T in g["T"]])
                    v = "" if np.isnan(bnd).all() or loss == "point" else f" viol {int((x.values > bnd + 1e-9).sum())}/{len(x)} (bound {bnd.max():.2f})"
                    cells.append(f"{loss} {ms(x)}{v}")
                P(f"   {dl} m={m}: " + " | ".join(cells))
        # paired fixed share - Hedge
        a, b = (r[(r.estimator == e) & (r.scenario == sc) & np.isclose(r.delta, d)].set_index("seed") for d in (1e-4, 0.0))
        P(f"   fixed share - Hedge (paired, = difference in cumulative MFF loss): mix {ms(a.loss_mix - b.loss_mix)} | point {ms(a.loss_point - b.loss_point)}")
        if sc == "jump":
            j = g0[g0.delta == 0]
            sw = j[j.sw_t > 0]
            P(f"   best one-switch sequence: {len(sw)}/{len(j)} runs switch; switch time median {sw.sw_t.median():.0f} "
              f"(IQR {sw.sw_t.quantile(.25):.0f}-{sw.sw_t.quantile(.75):.0f}; true change 2500); "
              f"arms: " + ", ".join(f"{a:g}->{b:g} x{n}" for (a, b), n in sw.groupby(['sw_from', 'sw_to']).size().sort_values(ascending=False).items()))
open(R + "regret_2026-11-14.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
