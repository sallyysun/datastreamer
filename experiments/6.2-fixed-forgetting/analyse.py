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
V_NMAX = float(os.environ.get("V_NMAX", "2000"))
V_NMIN = 50
MEMORY_LABELS = [(f"{m}.000000", str(m)) for m in C.MEMORIES] + [("nocum", "No forgetting")]
RECOVERY_RUN, RECOVERY_THRESHOLD = 20, 0.1
PRE_JUMP_FROM, TAIL = 2000, 500
PRE, POST = C.JUMP_RHO_S
GAP = abs(PRE - POST)
DRIFT_RATE = C.DRIFT_SPAN / (C.T - 1)   # d rho_S / dt; the drift path is linear in rho_S


def read_family(name, family):
    return pd.read_csv(R + f"{name}{C.sfx(family)}.csv", dtype={"neff": str})


def load(name):
    rows = pd.concat([read_family(name, f) for f in C.FAMILIES])
    rows["k"] = rows.neff.map(lambda v: "nocum" if v == "nocum" else str(round(float(v))))
    return rows


def rmse_summary(seeds):
    """Whole-window RMSE at each memory, the best memory, and no forgetting minus the longest memory."""
    whole = seeds[(seeds.grid == "main") & (seeds.window == "whole")]
    lines = ["Whole-window RMSE × 10⁻², mean (SE). Paired nocum − 1999 uses ±2 SE.",
             f"{'family':9s} {'estimator':16s} " + " ".join(f"{m:>11s}" for m in C.MEMORY_KEYS)
             + f" {'best':>6s} {'paired nocum-1999':>20s}"]
    for scenario in C.SCENARIOS:
        lines.extend(["", scenario])
        for family in C.FAMILIES:
            for estimator in C.ESTIMATORS:
                rows = whole[(whole.scenario == scenario) & (whole.copula == family) & (whole.estimator == estimator)]
                scores = rows.pivot(index="seed", columns="k", values="rmse")[C.MEMORY_KEYS] * 100
                mean, se = scores.mean(), standard_error(scores)
                paired = scores["nocum"] - scores["1999"]
                paired_mean, paired_se = paired.mean(), standard_error(paired)
                cells = " ".join(f"{mean[m]:6.2f}({se[m]:.2f})" for m in C.MEMORY_KEYS)
                lines.append(f"{C.FAMILY_TEXT[family]:9s} {C.EST_LABEL[estimator]:16s} " + cells
                             + f" {mean.idxmin():>6s} {paired_mean:+8.2f} ± {2 * paired_se:.2f} {win_loss_tie(paired_mean, paired_se)}")
    return lines


def log_quadratic_minimum(neff, mse, around):
    """Minimum of a quadratic in log N through the points in the slice around."""
    curve, slope, _ = np.polyfit(np.log(neff[around]), mse[around], 2)
    return np.exp(-slope / (2 * curve)), curve


def drift_optimum(decomposition, rate):
    """Log-quadratic optimum through the grid minimum and its neighbours, and the (2v/d^2)^(1/3) approximation."""
    lines = ["", f"Drift optimum vs (2v/d²)^(1/3) (d={rate:.5e}, v from {V_NMIN} ≤ N ≤ {V_NMAX:g})",
             f"{'family':9s} {'estimator':16s} {'grid min':>8s} {'bracket':>11s} {'optimum':>8s} {'5-pt':>6s} "
             f"{'approx':>8s} {'diff %':>7s} {'v':>6s} {'MSE×1e3':>8s}"]
    rows = []
    drift = decomposition[(decomposition.scenario == "drift") & (decomposition.window == "whole") & (decomposition.k != "nocum")].copy()
    drift["N"] = drift.neff.astype(float)
    for family in C.FAMILIES:
        for estimator in C.ESTIMATORS:
            g = drift[(drift.copula == family) & (drift.estimator == estimator)].sort_values("N").drop_duplicates("N")
            neff, mse = g.N.values, g.mse.values
            best = int(np.argmin(mse))
            optimum, curve = log_quadratic_minimum(neff, mse, slice(best - 1, best + 2))
            optimum_5pt, _ = log_quadratic_minimum(neff, mse, slice(best - 2, best + 3))
            bracketed = (0 < best < len(neff) - 1 and curve > 0 and neff[best - 1] <= optimum <= neff[best + 1]
                         and 200 <= neff[best] <= 600)
            sweep = g[(g.grid == "sweep") & (g.N >= V_NMIN) & (g.N <= V_NMAX)]
            v = np.median(sweep["var"].values * sweep.N.values)
            approx = max(1.0, (2 * v / rate ** 2) ** (1 / 3))
            diff_pct = 100 * (approx - optimum) / optimum
            rows.append(dict(copula=family, estimator=estimator, optimum=optimum, approx=approx, diff_pct=diff_pct, v=v, ok=bracketed))
            lines.append(f"{C.FAMILY_TEXT[family]:9s} {C.EST_LABEL[estimator]:16s} {neff[best]:8.0f} "
                         f"{neff[best - 1]:5.0f}-{neff[best + 1]:<5.0f} {optimum:8.1f} {optimum_5pt:6.0f} "
                         f"{approx:8.1f} {diff_pct:+7.2f} {v:6.3f} {1e3 * mse[best]:8.3f}{'' if bracketed else '  CHECK GRID'}")
    return lines, pd.DataFrame(rows)


def fixed_ess():
    optimum_lines, optimum = drift_optimum(load("fixed_ess_decomp"), DRIFT_RATE)
    full = V_NMAX == 2000
    optimum.to_csv(R + ("fixed_ess_optimum.csv" if full else f"fixed_ess_optimum_v{int(V_NMAX)}.csv"), index=False)
    if reports_enabled():
        lines = rmse_summary(load("fixed_ess_seeds")) + optimum_lines
        write_report(R + ("fixed_ess_2026-11-12.txt" if full else f"fixed_ess_v{int(V_NMAX)}_2026-11-14.txt"), lines)


def recovery_time(relative_bias):
    """First tick after the jump that starts RECOVERY_RUN consecutive forecasts within the threshold, or None."""
    within = relative_bias <= RECOVERY_THRESHOLD
    for k in range(len(relative_bias) - RECOVERY_RUN + 1):
        if within[k:k + RECOVERY_RUN].all():
            return k
    return None


def measure_recovery(include_checks):
    rows, checks = [], []
    for family in C.FAMILIES:
        bias = read_family("fixed_ess_jumpbias", family)
        if include_checks:
            matched = bias.merge(read_family("fixed_ess_jumppath", family), on=["copula", "estimator", "neff", "t"],
                                 suffixes=("", "_path"))
            max_difference = (abs(matched.mse - matched.mse_path) / matched.mse_path).max()
            checks.append(f"{family} {max_difference:.1e} ({len(matched)} rows)")
        for estimator in C.ESTIMATORS:
            for key, label in MEMORY_LABELS:
                arm = bias[(bias.estimator == estimator) & (bias.neff == key)]
                after = arm[arm.t >= C.CHANGE].sort_values("t")
                relative = np.abs(after.mean_err.values) / GAP
                rows.append(dict(copula=family, estimator=estimator, neff=label, recovery=recovery_time(relative),
                                 b_at_0=relative[0], b_at_100=relative[100],
                                 tail_B=np.abs(after.mean_err.values[-TAIL:]).mean() / GAP,
                                 pre_bias=arm[(arm.t >= PRE_JUMP_FROM) & (arm.t < C.CHANGE)].mean_err.mean()))
    return pd.DataFrame(rows), checks


def recovery_report(recovery, checks):
    lines = [f"Recovery after ρₛ {PRE:.4f} → {POST:.4f}: observations to relative bias ≤ {RECOVERY_THRESHOLD:g} for {RECOVERY_RUN} forecasts "
             "[bias at 100], tail = mean over the last 500.",
             f"Max relative MSE difference vs jump path · {', '.join(checks)}",
             "", f"{'N_eff':14s} " + "  ".join(f"{C.EST_LABEL[e]:>26s}" for e in C.ESTIMATORS)]
    for family in C.FAMILIES:
        lines.append(C.FAMILY_TEXT[family])
        for _, label in MEMORY_LABELS:
            cells = []
            for estimator in C.ESTIMATORS:
                r = recovery[(recovery.copula == family) & (recovery.estimator == estimator) & (recovery.neff == label)].iloc[0]
                time = "not recovered" if pd.isna(r.recovery) else f"{int(r.recovery)}"
                cells.append(f"{time:>13s} [{r.b_at_100:.2f}] tail {r.tail_B:.3f}")
            lines.append(f"{label:14s} " + "  ".join(f"{c:>26s}" for c in cells))
    return lines


def jump_recovery():
    include_report = reports_enabled()
    recovery, checks = measure_recovery(include_report)
    recovery.to_csv(R + "jump_recovery.csv", index=False)
    if include_report:
        write_report(R + "jump_recovery_2026-11-12.txt", recovery_report(recovery, checks))


def main():
    fixed_ess()
    jump_recovery()


if __name__ == "__main__":
    main()
