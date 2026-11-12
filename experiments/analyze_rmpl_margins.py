"""RMPL margin sensitivity (constant + drift, Gaussian copula, matched paths, 40 seeds 42-81).
Inputs: results/memory_sweep_matched_rmpl2.csv and memory_sweep_seeds_matched_rmpl2.csv (probe_memory with
MEM_FAMILIES=rmpl_ew,rmpl_ewcum,rmpl_exact,rmpl_kll MEM_SCENARIOS=constant,drift MEM_NOCUM=1 MEM_SEEDS_OUT=1
TAG=matched_rmpl2), results/rmpl_margins_2026-11-12.txt (probe_rmpl_margins).
Usage (from experiments/): python3 analyze_rmpl_margins.py -> results/rmpl_margin_sensitivity_2026-11-12.txt"""
import math
import numpy as np
import pandas as pd

R = "results/"
FAM = {"rmpl_ew": "EW 500", "rmpl_ewcum": "cumulative EW", "rmpl_exact": "oracle PIT", "rmpl_kll": "cum. KLL (ref.)"}
T = 5000
t = np.arange(500, T); r = 0.2 + 0.6 * t / (T - 1)
D = np.sqrt(np.mean(((3 / np.pi) / np.sqrt(1 - r ** 2 / 4) * 0.6 / (T - 1)) ** 2))
NOCUM = 1e9   # mem_ticks of the no-forgetting arm is ~2e12

agg = pd.read_csv(R + "memory_sweep_matched_rmpl2.csv")
sd = pd.read_csv(R + "memory_sweep_seeds_matched_rmpl2.csv")
agg = agg[agg.window == "all"]; sd = sd[sd.window == "all"]
L = []
P = lambda s="": L.append(s)


def opt(g):
    g = g[g.mem_ticks < NOCUM].sort_values("mem_ticks")
    N, M = g.mem_ticks.values, g.mse.values
    i = int(np.argmin(M))
    if i == 0 or i == len(N) - 1:
        return N[i], None, i, g
    c2, c1, _ = np.polyfit(np.log(N[i - 1:i + 2]), M[i - 1:i + 2], 2)
    return np.exp(-c1 / (2 * c2)), c2, i, g


def paired(a, b):
    d = (a - b).dropna()
    return d.mean(), d.std(ddof=1) / math.sqrt(len(d))


P("RMPL marginal sensitivity: Gaussian copula, matched paths, 40 seeds (42-81), 20 dependence memories 25-30000 plus")
P("no forgetting; estimate after each update; window t = 500-4999; Linux/libstdc++. RMSE x 10^-2.")
P("cum. KLL is a secondary reference only (it also changes the approximation, and its DataSketches coin is shared")
P("across the probe's threads, so it varies slightly run to run).")
P()
for sc in ["constant", "drift"]:
    P(f"=== {sc} ===")
    P(f"{'margin':16s} {'grid-min N':>10s} {'RMSE@min':>9s} {'RMSE@1999':>10s} {'RMSE nocum':>10s} {'optimum':>8s} {'predicted':>9s} {'diff%':>7s}")
    for f in FAM:
        g = agg[(agg.scenario == sc) & (agg.estimator == f)]
        nm, c2, i, gg = opt(g)
        r1999 = 100 * gg.iloc[(gg.mem_ticks - 1999).abs().argmin()].rmse
        nc = 100 * g[g.mem_ticks > NOCUM].rmse.iloc[0]
        line = f"{FAM[f]:16s} {gg.mem_ticks.values[i]:10.0f} {100*gg.rmse.values[i]:9.2f} {r1999:10.2f} {nc:10.2f}"
        if sc == "drift" and c2 is not None:
            N = gg.mem_ticks.values; w = (N >= 50) & (N <= 2000)
            s2 = np.median((gg["var"].values * N)[w]); npred = (2 * s2 / D ** 2) ** (1 / 3)
            line += f" {nm:8.1f} {npred:9.1f} {100*(npred-nm)/nm:+7.2f}"
        elif c2 is None:
            line += f" {'edge':>8s}"
        P(line)
    P()
    P("Paired RMSE difference vs oracle PIT on the same streams (x 10^-2, mean ± 2 SE), and mean signed error (x 10^-2):")
    mems = sorted(sd[(sd.scenario == sc) & (sd.mem_ticks < NOCUM)].mem_ticks.unique())
    pick = [mems[k] for k in (2, 6, 9, 10, 12, 14)] + [max(sd.mem_ticks)]
    hdr = "  ".join(("nocum" if m > NOCUM else f"N{m:.0f}").rjust(15) for m in pick)
    P(f"{'':16s} {hdr}")
    for f in ["rmpl_ew", "rmpl_ewcum", "rmpl_kll"]:
        cells = []
        for m in pick:
            a = sd[(sd.scenario == sc) & (sd.estimator == f) & np.isclose(sd.mem_ticks, m)].set_index("seed").rmse
            b = sd[(sd.scenario == sc) & (sd.estimator == "rmpl_exact") & np.isclose(sd.mem_ticks, m)].set_index("seed").rmse
            mu, se = paired(100 * a, 100 * b)
            cells.append(f"{mu:+6.2f}±{2*se:4.2f}")
        P(f"{FAM[f]:16s} " + "  ".join(c.rjust(15) for c in cells))
    for f in FAM:
        cells = [f"{100*sd[(sd.scenario==sc)&(sd.estimator==f)&np.isclose(sd.mem_ticks,m)].mean_err.mean():+6.2f}" for m in pick]
        P(f"{'err ' + FAM[f]:16s} " + "  ".join(c.rjust(15) for c in cells))
    P()

# Mechanism check: the Gaussian score's fixed point with the measured normal-score moments.
P("Mechanism check: fixed point of E[score] = 0, r(1-r^2) + (1+r^2) E[z1 z2] - 2 r E[z^2] = 0, using the measured")
P("normal-score moments of each margin (constant path, true r = 0.6, rho_S = 0.582); compared with the measured mean")
P("signed error of the no-forgetting arm under constant dependence (the arm with no lag and the least noise).")
mom = {"rmpl_ew": (0.5629, 0.9454), "rmpl_ewcum": (0.5878, 0.9850), "rmpl_exact": (0.5929, 0.9916)}   # rmpl_margins txt
for f, (Pz, Qz) in mom.items():
    rr = np.roots([-1, Pz, 1 - 2 * Qz, Pz]); rr = rr[np.isreal(rr)].real; rr = rr[(rr > -1) & (rr < 1)][0]
    rs = 6 / np.pi * np.arcsin(rr / 2)
    me = 100 * sd[(sd.scenario == "constant") & (sd.estimator == f) & (sd.mem_ticks > NOCUM)].mean_err.mean()
    P(f"  {FAM[f]:14s} predicted rho_S {rs:.4f} -> bias {100*(rs-0.582):+.2f} x 10^-2; measured nocum mean error {me:+.2f} x 10^-2")
open(R + "rmpl_margin_sensitivity_2026-11-12.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
