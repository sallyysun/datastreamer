"""Section 6.2 recovery after the abrupt jump: for each fixed N_eff, the number of post-jump observations until the
mean estimate stays within 10% of the jump size for 20 consecutive forecasts. Writes results/jump_recovery.csv and
results/jump_recovery_2026-11-12.txt."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

R = "results/"
FAMS = [(f, C.sfx(f)) for f in C.FAMILIES]
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
MEM = [(f"{m}.000000", str(m)) for m in C.MEMORIES] + [("nocum", "No forgetting")]
RUN, THR = 20, 0.1
PRE, POST = C.rho_s(C.JUMP_KNOB[0]), C.rho_s(C.JUMP_KNOB[1])
GAP = abs(PRE - POST)


def recovery(b):
    ok = b <= THR
    for k in range(len(b) - RUN + 1):
        if ok[k:k + RUN].all():
            return k
    return None


def measure():
    rows, checks = [], []
    for f, s in FAMS:
        jb = pd.read_csv(R + f"fixed_ess_jumpbias{s}.csv", dtype={"neff": str})
        jp = pd.read_csv(R + f"fixed_ess_jumppath{s}.csv", dtype={"neff": str})
        m = jb.merge(jp, on=["copula", "estimator", "neff", "t"], suffixes=("", "_old"))
        checks.append(f"check {f}: per-tick MSE vs fixed_ess_jumppath: {len(m)} rows, max relative difference "
                      f"{(abs(m.mse - m.mse_old) / m.mse_old).max():.1e}")
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
    out = [f"90% recovery after the jump: rho_S {PRE:.4f} -> {POST:.4f} (gap {GAP:.4f}); threshold B(k) <= {THR} held for {RUN}",
           "consecutive forecasts; k = post-jump observations absorbed before the forecast; 60 seeds; Linux/libstdc++."]
    out += checks + [""]
    for f, _ in FAMS:
        out.append(f"=== {f}: recovery time (observations); B(100) in brackets; tail = mean B over the last 500 observations ===")
        out.append(f"{'N_eff':14s} " + "  ".join(f"{name:>26s}" for _, name in EST))
        for _, lab in MEM:
            cells = []
            for e, _ in EST:
                r = df[(df.copula == f) & (df.estimator == e) & (df.neff == lab)].iloc[0]
                rec = "not recovered" if pd.isna(r.recovery) else f"{int(r.recovery)}"
                cells.append(f"{rec:>13s} [{r.b_at_100:.2f}] tail {r.tail_B:.3f}")
            out.append(f"{lab:14s} " + "  ".join(f"{c:>26s}" for c in cells))
        out.append("")
    return out


def main():
    df, checks = measure()
    df.to_csv(R + "jump_recovery.csv", index=False)
    lines = report(df, checks)
    open(R + "jump_recovery_2026-11-12.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
