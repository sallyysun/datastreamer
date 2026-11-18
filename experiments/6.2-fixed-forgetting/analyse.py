"""Section 6.2 tables: fixed-memory RMSE, the drift optimum against its approximation, and recovery after the jump."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import reports_enabled, write_report  # noqa: E402
from summary_stats import standard_error, win_loss_tie  # noqa: E402
import config as C  # noqa: E402

R = "results/"
VMAX = float(os.environ.get("V_NMAX", "2000"))
MEM = [(f"{m}.000000", str(m)) for m in C.MEMORIES] + [("nocum", "No forgetting")]
RUN, THR = 20, 0.1
PRE, POST = C.JUMP_RHO_S
GAP = abs(PRE - POST)


def read_family(name, f):
    return pd.read_csv(R + f"{name}{C.sfx(f)}.csv", dtype={"neff": str})


def load(name):
    d = pd.concat([read_family(name, f) for f in C.FAMILIES])
    d["k"] = d.neff.map(lambda v: "nocum" if v == "nocum" else str(round(float(v))))
    return d


def drift_rate():
    """d rho_S / dt along the drift path, which is linear in rho_S."""
    return C.DRIFT_SPAN / (C.T - 1)


def rmse_summary(seeds):
    whole = seeds[(seeds.grid == "main") & (seeds.window == "whole")]
    lines = ["Whole-window RMSE × 10⁻², mean (SE). Paired nocum − 1999 uses ±2 SE.",
             f"{'family':9s} {'estimator':16s} " + " ".join(f"{m:>11s}" for m in C.MEMORY_KEYS)
             + f" {'best':>6s} {'paired nocum-1999':>20s}"]
    for scenario in ("constant", "drift", "jump"):
        lines.extend(["", scenario])
        for family in C.FAMILIES:
            for estimator in C.ESTIMATORS:
                rows = whole[(whole.scenario == scenario) & (whole.copula == family) & (whole.estimator == estimator)]
                scores = rows.pivot(index="seed", columns="k", values="rmse")[C.MEMORY_KEYS] * 100
                mean, se = scores.mean(), standard_error(scores)
                paired = scores["nocum"] - scores["1999"]
                paired_mean, paired_se = paired.mean(), standard_error(paired)
                verdict = win_loss_tie(paired_mean, paired_se)
                lines.append(f"{C.FAMILY_TEXT[family]:9s} {C.EST_LABEL[estimator]:16s} "
                             + " ".join(f"{mean[m]:6.2f}({se[m]:.2f})" for m in C.MEMORY_KEYS)
                             + f" {mean.idxmin():>6s} {paired_mean:+8.2f} ± {2 * paired_se:.2f} {verdict}")
    return lines


def drift_optimum(dec, D):
    """Log-quadratic optimum through the grid minimum and its neighbours, and the (2v/d^2)^(1/3) approximation."""
    lines = ["", f"Drift optimum vs (2v/d²)^(1/3) (d={D:.5e}, v from 50 ≤ N ≤ {VMAX:g})",
             f"{'family':9s} {'estimator':16s} {'grid min':>8s} {'bracket':>11s} {'optimum':>8s} {'5-pt':>6s} {'approx':>8s} {'diff %':>7s} {'v':>6s} {'MSE×1e3':>8s}"]
    rows = []
    dd = dec[(dec.scenario == "drift") & (dec.window == "whole") & (dec.k != "nocum")].copy()
    dd["N"] = dd.neff.astype(float)
    for f in C.FAMILIES:
        for e in C.ESTIMATORS:
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
            lines.append(f"{C.FAMILY_TEXT[f]:9s} {C.EST_LABEL[e]:16s} {N[i]:8.0f} {N[i-1]:5.0f}-{N[i+1]:<5.0f} {nm:8.1f} {n5:6.0f} "
                         f"{npred:8.1f} {100*(npred-nm)/nm:+7.2f} {v:6.3f} {1e3*M[i]:8.3f}"
                         f"{'' if ok else '  CHECK GRID'}")
    return lines, pd.DataFrame(rows)


def fixed_ess():
    opt_lines, opt = drift_optimum(load("fixed_ess_decomp"), drift_rate())
    full = VMAX == 2000
    opt.to_csv(R + ("fixed_ess_optimum.csv" if full else f"fixed_ess_optimum_v{int(VMAX)}.csv"), index=False)
    if reports_enabled():
        lines = rmse_summary(load("fixed_ess_seeds")) + opt_lines
        write_report(R + ("fixed_ess_2026-11-12.txt" if full else f"fixed_ess_v{int(VMAX)}_2026-11-14.txt"), lines)


def recovery(b):
    ok = b <= THR
    for k in range(len(b) - RUN + 1):
        if ok[k:k + RUN].all():
            return k
    return None


def measure_recovery(include_checks):
    rows, checks = [], []
    for f in C.FAMILIES:
        jb = read_family("fixed_ess_jumpbias", f)
        if include_checks:
            matched = jb.merge(read_family("fixed_ess_jumppath", f), on=["copula", "estimator", "neff", "t"],
                               suffixes=("", "_path"))
            max_difference = (abs(matched.mse - matched.mse_path) / matched.mse_path).max()
            checks.append(f"{f} {max_difference:.1e} ({len(matched)} rows)")
        for e in C.ESTIMATORS:
            for k, lab in MEM:
                arm = jb[(jb.estimator == e) & (jb.neff == k)]
                g = arm[arm.t >= C.CHANGE].sort_values("t")
                b = np.abs(g.mean_err.values) / GAP
                rows.append(dict(copula=f, estimator=e, neff=lab, recovery=recovery(b), b_at_0=b[0], b_at_100=b[100],
                                 tail_B=np.abs(g.mean_err.values[-500:]).mean() / GAP,
                                 pre_bias=arm[(arm.t >= 2000) & (arm.t < C.CHANGE)].mean_err.mean()))
    return pd.DataFrame(rows), checks


def recovery_report(df, checks):
    lines = [f"Recovery after ρₛ {PRE:.4f} → {POST:.4f}: observations to relative bias ≤ {THR:g} for {RUN} forecasts "
             "[bias at 100], tail = mean over the last 500.",
             f"Max relative MSE difference vs jump path · {', '.join(checks)}",
             "", f"{'N_eff':14s} " + "  ".join(f"{C.EST_LABEL[e]:>26s}" for e in C.ESTIMATORS)]
    for f in C.FAMILIES:
        lines.append(C.FAMILY_TEXT[f])
        for _, lab in MEM:
            cells = []
            for e in C.ESTIMATORS:
                r = df[(df.copula == f) & (df.estimator == e) & (df.neff == lab)].iloc[0]
                rec = "not recovered" if pd.isna(r.recovery) else f"{int(r.recovery)}"
                cells.append(f"{rec:>13s} [{r.b_at_100:.2f}] tail {r.tail_B:.3f}")
            lines.append(f"{lab:14s} " + "  ".join(f"{c:>26s}" for c in cells))
    return lines


def jump_recovery():
    include_report = reports_enabled()
    df, checks = measure_recovery(include_report)
    df.to_csv(R + "jump_recovery.csv", index=False)
    if include_report:
        write_report(R + "jump_recovery_2026-11-12.txt", recovery_report(df, checks))


def main():
    fixed_ess()
    jump_recovery()


if __name__ == "__main__":
    main()
