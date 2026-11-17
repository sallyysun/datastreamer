"""Chapter 6.2 fixed-forgetting analysis from results/fixed_ess_{seeds,decomp}[_fam].csv (probe_fixed_ess.cpp).
Main comparison (constant / drift / jump windows), drift fine-grid optimum (log-quadratic through the grid minimum and
its two neighbours, on the union of the main, fine and sweep grids), Proposition prop:optimal-ess approximation
N* = (2 v / d^2)^(1/3): v = median of var * N_eff over the sweep-grid memories with 50 <= N_eff <= 2000 (the established
procedure), d = RMS of d rho_S / dt along the matched drift path. Percentages from unrounded values.
Usage: python3 analyze_fixed_ess.py -> results/fixed_ess_2026-11-12.txt, results/fixed_ess_optimum.csv"""
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

R = "results/"
FAMS = [(f, C.sfx(f)) for f in C.FAMILIES]
FL = C.FAMILY_TEXT
EL = {e: C.EST_LABEL[e] for e in ("xiao", "hermite", "rmpl")}
MAIN = C.MEMORY_KEYS
T = C.T
t = np.arange(C.BURN, T); r = C.DRIFT_FROM + C.DRIFT_SPAN * t / (T - 1)
D = np.sqrt(np.mean(((3 / np.pi) / np.sqrt(1 - r ** 2 / 4) * C.DRIFT_SPAN / (T - 1)) ** 2))
VMAX = float(os.environ.get("V_NMAX", "2000"))   # upper end of the N_eff range for v; 350 = the flat region (calibrated v)

seeds = pd.concat([pd.read_csv(R + f"fixed_ess_seeds{s}.csv", dtype={"neff": str}) for _, s in FAMS])
dec = pd.concat([pd.read_csv(R + f"fixed_ess_decomp{s}.csv", dtype={"neff": str}) for _, s in FAMS])
key = lambda v: "nocum" if v == "nocum" else str(round(float(v)))
seeds["k"] = seeds.neff.map(key); dec["k"] = dec.neff.map(key)
mainS = seeds[seeds.grid == "main"]
L = []; P = lambda s="": L.append(s)


def cell(sc, w, f, e):
    p = mainS[(mainS.scenario == sc) & (mainS.window == w) & (mainS.copula == f) & (mainS.estimator == e)]
    return p.pivot(index="seed", columns="k", values="rmse") * 100


P("Chapter 6.2 fixed forgetting. 60 seeds (42-101), matched paths, forecast held before each tick, Linux/libstdc++.")
P("RMPL margins: cumulative DataSketches KLL k = 256, one pseudo-observation sequence per stream shared by every arm.")
P("RMSE x 10^-2, mean (SE). 'paired' = no forgetting - N_eff 1999, mean ± 2 SE; W/L/T at 2 SE.")
for sc, wins in [("constant", ["whole"]), ("drift", ["whole"]), ("jump", ["pre", "rec_0_100", "rec_100_500", "post", "whole"])]:
    for w in wins:
        P(); P(f"=== {sc}, window {w} ===")
        P(f"{'family':9s} {'estimator':16s} " + " ".join(f"{m:>11s}" for m in MAIN) + f" {'best':>6s} {'paired nocum-1999':>20s}")
        for f, _ in FAMS:
            for e in EL:
                p = cell(sc, w, f, e)
                mu = p[MAIN].mean(); se = p[MAIN].std(ddof=1) / math.sqrt(len(p))
                d = p["nocum"] - p["1999"]; dm, ds = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
                v = "W" if dm < -2 * ds else "L" if dm > 2 * ds else "T"
                P(f"{FL[f]:9s} {EL[e]:16s} " + " ".join(f"{mu[m]:6.2f}({se[m]:.2f})" for m in MAIN)
                  + f" {mu.idxmin():>6s} {dm:+8.2f} ± {2*ds:.2f} {v}")

# --- drift: fine-grid optimum and the approximation
P(); P("=== drift: empirical MSE-optimal N_eff (fine + main + sweep grids) vs (2v/d^2)^(1/3) ===")
P(f"d_rms = {D:.5e}. v from sweep-grid memories 50 <= N <= {VMAX:g} (median of var * N). optimum: log-quadratic through the")
P("grid minimum and its two neighbours; 'bracket' = those neighbours; 5-pt = the same through five points (sensitivity).")
P(f"{'family':9s} {'estimator':16s} {'grid min':>8s} {'bracket':>11s} {'optimum':>8s} {'5-pt':>6s} {'approx':>8s} {'diff %':>7s} {'v':>6s} {'MSE min':>8s}")
rows = []
dd = dec[(dec.scenario == "drift") & (dec.window == "whole") & (dec.k != "nocum")].copy()
dd["N"] = dd.neff.astype(float)
for f, _ in FAMS:
    for e in EL:
        g = dd[(dd.copula == f) & (dd.estimator == e)].sort_values("N").drop_duplicates("N")
        N, M = g.N.values, g.mse.values
        i = int(np.argmin(M)); ok = 0 < i < len(N) - 1
        c2, c1, _ = np.polyfit(np.log(N[i - 1:i + 2]), M[i - 1:i + 2], 2); nm = np.exp(-c1 / (2 * c2))
        c5 = np.polyfit(np.log(N[i - 2:i + 3]), M[i - 2:i + 3], 2); n5 = np.exp(-c5[1] / (2 * c5[0]))
        ok = ok and c2 > 0 and N[i - 1] <= nm <= N[i + 1] and 200 <= N[i] <= 600
        sw = g[(g.grid == "sweep") & (g.N >= 50) & (g.N <= VMAX)]
        v = np.median(sw["var"].values * sw.N.values); npred = max(1.0, (2 * v / D ** 2) ** (1 / 3))
        rows.append(dict(copula=f, estimator=e, optimum=nm, approx=npred, diff_pct=100 * (npred - nm) / nm, v=v, ok=ok))
        P(f"{FL[f]:9s} {EL[e]:16s} {N[i]:8.0f} {N[i-1]:5.0f}-{N[i+1]:<5.0f} {nm:8.1f} {n5:6.0f} {npred:8.1f} {100*(npred-nm)/nm:+7.2f} "
          f"{v:6.3f} {1e3*M[i]:8.3f}  {'interior, convex, inside fine grid' if ok else 'CHECK GRID'}")
pd.DataFrame(rows).to_csv(R + ("fixed_ess_optimum.csv" if VMAX == 2000 else f"fixed_ess_optimum_v{int(VMAX)}.csv"), index=False)

open(R + ("fixed_ess_2026-11-12.txt" if VMAX == 2000 else f"fixed_ess_v{int(VMAX)}_2026-11-14.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
