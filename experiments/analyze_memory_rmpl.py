"""Optimal memory under Gaussian drift for Xiao, Hermite and RMPL (three margin variants), from
results/memory_sweep_matched_rmpl.csv (probe_memory, SCENARIOS=matched TAG=matched_rmpl MEM_SCENARIOS=drift
MEM_FAMILIES=xiao_ew,hermite_ew,rmpl_ew,rmpl_kll,rmpl_exact). Same procedure as plot_ch6_core.e2: measured optimum by
log-quadratic interpolation through the grid minimum and its neighbours; predicted (2 sigma^2 / d^2)^(1/3) with sigma^2 the
median of var * N_eff over 50 <= N_eff <= 2000 and d the RMS of d rho_S / dt along the matched path.
Usage (from experiments/): python3 analyze_memory_rmpl.py  -> results/memory_sweep_rmpl_2026-11-12.txt"""
import numpy as np
import pandas as pd

T = 5000
t = np.arange(500, T); r = 0.2 + 0.6 * t / (T - 1)
D = np.sqrt(np.mean(((3 / np.pi) / np.sqrt(1 - r ** 2 / 4) * 0.6 / (T - 1)) ** 2))
LAB = {"xiao_ew": "Counting matrix", "hermite_ew": "Hermite series", "rmpl_ew": "RMPL, EW margins (N_eff 500)",
       "rmpl_kll": "RMPL, cumulative KLL margins (k 256)", "rmpl_exact": "RMPL, exact PIT (oracle)"}


def optimum(N, M, half):
    i = int(np.argmin(M))
    c2, c1, _ = np.polyfit(np.log(N[i - half:i + half + 1]), M[i - half:i + half + 1], 2)
    return np.exp(-c1 / (2 * c2)), c2, i


d = pd.read_csv("results/memory_sweep_matched_rmpl.csv")
d = d[(d.scenario == "drift") & (d.window == "all")]
old = pd.read_csv("results/memory_sweep_matched.csv")
chk = d.merge(old, on=["estimator", "scenario", "window", "param"], suffixes=("", "_o"))
lines = ["Optimal effective sample size under Gaussian gradual drift (matched path), 40 seeds (42-81), 20 memories 25-30000,",
         "estimate recorded after each update, window t = 500-4999. Linux/libstdc++.",
         f"Check: Xiao/Hermite rows reproduce memory_sweep_matched.csv: {len(chk)} rows, max |d MSE| = {(chk.mse - chk.mse_o).abs().max():.1e}.",
         f"d_rms = {D:.4e} per tick.", "",
         f"{'estimator':40s} {'measured':>9s} {'5-pt':>6s} {'bracket':>11s} {'predicted':>9s} {'diff %':>7s} {'sigma2':>7s} {'RMSE min':>9s}"]
for e in LAB:
    g = d[d.estimator == e].sort_values("mem_ticks")
    N, M = g.mem_ticks.values, g.mse.values
    nm, c2, i = optimum(N, M, 1)
    n5, _, _ = optimum(N, M, 2)
    w = (N >= 50) & (N <= 2000)
    s2 = np.median((g["var"].values * N)[w])
    npred = (2 * s2 / D ** 2) ** (1 / 3)
    ok = c2 > 0 and N[i - 1] <= nm <= N[i + 1] and 0 < i < len(N) - 1
    lines.append(f"{LAB[e]:40s} {nm:9.1f} {n5:6.0f} {N[i-1]:5.0f}-{N[i+1]:<5.0f} {npred:9.1f} {100*(npred-nm)/nm:+7.2f} {s2:7.3f} "
                 f"{100*np.sqrt(M[i]):7.2f}e-2  {'interior, convex' if ok else 'CHECK'}")
lines += ["", "diff % = (predicted - measured) / measured, unrounded. 5-pt: the same interpolation through five points (sensitivity)."]
open("results/memory_sweep_rmpl_2026-11-12.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
