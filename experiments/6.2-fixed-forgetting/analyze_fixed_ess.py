"""Section 6.2 fixed forgetting: RMSE of each fixed N_eff per scenario window, and the drift-optimal N_eff against
the approximation N* = (2 v / d^2)^(1/3). Writes results/fixed_ess_2026-11-12.txt and results/fixed_ess_optimum.csv
(V_NMAX=350: v from N_eff <= 350 only, written as *_v350*)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error, win_loss_tie  # noqa: E402
import config as C  # noqa: E402

R = "results/"
FAMS = [(f, C.sfx(f)) for f in C.FAMILIES]
EST = ["xiao", "hermite", "rmpl"]
WINDOWS = [("constant", ["whole"]), ("drift", ["whole"]), ("jump", ["pre", "rec_0_100", "rec_100_500", "post", "whole"])]
VMAX = float(os.environ.get("V_NMAX", "2000"))


def drift_rate():
    """d rho_S / dt along the drift path, which is linear in rho_S."""
    return C.DRIFT_SPAN / (C.T - 1)


def load(name):
    d = pd.concat([pd.read_csv(R + f"{name}{s}.csv", dtype={"neff": str}) for _, s in FAMS])
    d["k"] = d.neff.map(lambda v: "nocum" if v == "nocum" else str(round(float(v))))
    return d


def window_tables(seeds):
    main = seeds[seeds.grid == "main"]
    out = ["Chapter 6.2 fixed forgetting. 60 seeds (42-101), matched paths, forecast held before each tick, Linux/libstdc++.",
           "RMPL margins: cumulative DataSketches KLL k = 256, one pseudo-observation sequence per stream shared by every arm.",
           "RMSE x 10^-2, mean (SE). 'paired' = no forgetting - N_eff 1999, mean ± 2 SE; W/L/T at 2 SE.",
           "", f"{'family':9s} {'estimator':16s} " + " ".join(f"{m:>11s}" for m in C.MEMORY_KEYS)
           + f" {'best':>6s} {'paired nocum-1999':>20s}"]
    for sc, wins in WINDOWS:
        for w in wins:
            out += ["", f"{sc} / {w}"]
            for f, _ in FAMS:
                for e in EST:
                    p = main[(main.scenario == sc) & (main.window == w) & (main.copula == f) & (main.estimator == e)]
                    p = p.pivot(index="seed", columns="k", values="rmse") * 100
                    scores = p[C.MEMORY_KEYS]
                    mu, se = scores.mean(), standard_error(scores)
                    d = p["nocum"] - p["1999"]
                    dm, ds = d.mean(), standard_error(d)
                    v = win_loss_tie(dm, ds)
                    out.append(f"{C.FAMILY_TEXT[f]:9s} {C.EST_LABEL[e]:16s} "
                               + " ".join(f"{mu[m]:6.2f}({se[m]:.2f})" for m in C.MEMORY_KEYS)
                               + f" {mu.idxmin():>6s} {dm:+8.2f} ± {2*ds:.2f} {v}")
    return out


def drift_optimum(dec, D):
    """Log-quadratic optimum through the grid minimum and its neighbours, and the (2v/d^2)^(1/3) approximation."""
    out = ["", "Optimal drift memory vs (2v/d^2)^(1/3)",
           f"d_rms = {D:.5e}. v from sweep-grid memories 50 <= N <= {VMAX:g} (median of var * N). optimum: log-quadratic through the",
           "grid minimum and its two neighbours; 'bracket' = those neighbours; 5-pt = the same through five points (sensitivity).",
           f"{'family':9s} {'estimator':16s} {'grid min':>8s} {'bracket':>11s} {'optimum':>8s} {'5-pt':>6s} {'approx':>8s} {'diff %':>7s} {'v':>6s} {'MSE min':>8s}"]
    rows = []
    dd = dec[(dec.scenario == "drift") & (dec.window == "whole") & (dec.k != "nocum")].copy()
    dd["N"] = dd.neff.astype(float)
    for f, _ in FAMS:
        for e in EST:
            g = dd[(dd.copula == f) & (dd.estimator == e)].sort_values("N").drop_duplicates("N")
            N, M = g.N.values, g.mse.values
            i = int(np.argmin(M))
            c2, c1, _ = np.polyfit(np.log(N[i - 1:i + 2]), M[i - 1:i + 2], 2)
            nm = np.exp(-c1 / (2 * c2))
            c5 = np.polyfit(np.log(N[i - 2:i + 3]), M[i - 2:i + 3], 2)
            n5 = np.exp(-c5[1] / (2 * c5[0]))
            ok = 0 < i < len(N) - 1 and c2 > 0 and N[i - 1] <= nm <= N[i + 1] and 200 <= N[i] <= 600
            sw = g[(g.grid == "sweep") & (g.N >= 50) & (g.N <= VMAX)]
            v = np.median(sw["var"].values * sw.N.values)
            npred = max(1.0, (2 * v / D ** 2) ** (1 / 3))
            rows.append(dict(copula=f, estimator=e, optimum=nm, approx=npred, diff_pct=100 * (npred - nm) / nm, v=v, ok=ok))
            out.append(f"{C.FAMILY_TEXT[f]:9s} {C.EST_LABEL[e]:16s} {N[i]:8.0f} {N[i-1]:5.0f}-{N[i+1]:<5.0f} {nm:8.1f} {n5:6.0f} "
                       f"{npred:8.1f} {100*(npred-nm)/nm:+7.2f} {v:6.3f} {1e3*M[i]:8.3f}  "
                       f"{'interior, convex, inside fine grid' if ok else 'CHECK GRID'}")
    return out, pd.DataFrame(rows)


def main():
    opt_lines, opt = drift_optimum(load("fixed_ess_decomp"), drift_rate())
    full = VMAX == 2000
    opt.to_csv(R + ("fixed_ess_optimum.csv" if full else f"fixed_ess_optimum_v{int(VMAX)}.csv"), index=False)
    if reports_enabled():
        lines = window_tables(load("fixed_ess_seeds")) + opt_lines
        write_report(R + ("fixed_ess_2026-11-12.txt" if full else f"fixed_ess_v{int(VMAX)}_2026-11-14.txt"), lines)


if __name__ == "__main__":
    main()
