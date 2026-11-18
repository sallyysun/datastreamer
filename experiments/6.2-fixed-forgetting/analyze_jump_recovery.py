"""Measure how quickly fixed-memory forecasts recover after the jump."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import reports_enabled, write_report  # noqa: E402
import config as C  # noqa: E402

R = "results/"
FAMS = [(f, C.sfx(f)) for f in C.FAMILIES]
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
MEM = [(f"{m}.000000", str(m)) for m in C.MEMORIES] + [("nocum", "No forgetting")]
RUN, THR = 20, 0.1
PRE, POST = C.JUMP_RHO_S
GAP = abs(PRE - POST)


def recovery(b):
    ok = b <= THR
    for k in range(len(b) - RUN + 1):
        if ok[k:k + RUN].all():
            return k
    return None


def measure(include_checks):
    rows, checks = [], []
    for f, s in FAMS:
        jb = pd.read_csv(R + f"fixed_ess_jumpbias{s}.csv", dtype={"neff": str})
        if include_checks:
            jump_path = pd.read_csv(R + f"fixed_ess_jumppath{s}.csv", dtype={"neff": str})
            matched = jb.merge(jump_path, on=["copula", "estimator", "neff", "t"], suffixes=("", "_path"))
            max_difference = (abs(matched.mse - matched.mse_path) / matched.mse_path).max()
            checks.append(f"{f} {max_difference:.1e} ({len(matched)} rows)")
        for e, _ in EST:
            for k, lab in MEM:
                arm = jb[(jb.estimator == e) & (jb.neff == k)]
                g = arm[arm.t >= C.CHANGE].sort_values("t")
                b = np.abs(g.mean_err.values) / GAP
                rows.append(dict(copula=f, estimator=e, neff=lab, recovery=recovery(b), b_at_0=b[0], b_at_100=b[100],
                                 tail_B=np.abs(g.mean_err.values[-500:]).mean() / GAP,
                                 pre_bias=arm[(arm.t >= 2000) & (arm.t < C.CHANGE)].mean_err.mean()))
    return pd.DataFrame(rows), checks


def report(df, checks):
    lines = [f"Recovery after ρₛ {PRE:.4f} → {POST:.4f} (relative bias ≤ {THR:g} for {RUN} forecasts)",
             "Cells show observations to recovery [relative bias at 100] and mean relative bias over the last 500.",
             f"Max relative MSE difference vs jump path · {', '.join(checks)}",
             "", f"{'N_eff':14s} " + "  ".join(f"{name:>26s}" for _, name in EST)]
    for f, _ in FAMS:
        lines.append(C.FAMILY_TEXT[f])
        for _, lab in MEM:
            cells = []
            for e, _ in EST:
                r = df[(df.copula == f) & (df.estimator == e) & (df.neff == lab)].iloc[0]
                rec = "not recovered" if pd.isna(r.recovery) else f"{int(r.recovery)}"
                cells.append(f"{rec:>13s} [{r.b_at_100:.2f}] tail {r.tail_B:.3f}")
            lines.append(f"{lab:14s} " + "  ".join(f"{c:>26s}" for c in cells))
    return lines


def main():
    include_report = reports_enabled()
    df, checks = measure(include_report)
    df.to_csv(R + "jump_recovery.csv", index=False)
    if include_report:
        write_report(R + "jump_recovery_2026-11-12.txt", report(df, checks))


if __name__ == "__main__":
    main()
