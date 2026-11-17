"""90% recovery time after the abrupt jump (Section 6.2), from results/fixed_ess_jumpbias[_fam].csv (probe_fixed_ess
JUMPBIAS=1: the same 60 seeds, streams and arms as fixed_ess_jumppath; checked here against its per-tick MSE).
Mean prequential estimate at post-jump step k (forecast held before observation 2500 + k, i.e. after k post-jump
observations) = rho_S,post + mean error. B(k) = |mean estimate - rho_S,post| / |rho_S,pre - rho_S,post|.
Recovery time = smallest k with B(j) <= 0.1 for j = k, ..., k + 19 (20 consecutive forecasts). Window: k <= 2499, so a
recovery must start by k = 2480; otherwise 'not recovered'.
Usage: python3 analyze_jump_recovery.py -> results/jump_recovery_2026-11-12.txt, results/jump_recovery.csv"""
import numpy as np
import pandas as pd

R = "results/"
FAMS = [("gaussian", ""), ("clayton", "_clayton"), ("gumbel", "_gumbel"), ("t", "_t")]
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
MEM = [("39.000000", "39"), ("99.000000", "99"), ("199.000000", "199"), ("399.000000", "399"), ("1999.000000", "1999"),
       ("nocum", "No forgetting")]
JUMP, N, RUN, THR = 2500, 5000, 20, 0.1
PRE, POST = 6 / np.pi * np.arcsin(0.1), 6 / np.pi * np.arcsin(0.4)
GAP = abs(PRE - POST)
L = []; P = lambda s="": L.append(s)


def recovery(b):
    ok = b <= THR
    for k in range(len(b) - RUN + 1):
        if ok[k:k + RUN].all():
            return k
    return None


P(f"90% recovery after the jump: rho_S {PRE:.4f} -> {POST:.4f} (gap {GAP:.4f}); threshold B(k) <= {THR} held for {RUN}")
P("consecutive forecasts; k = post-jump observations absorbed before the forecast; 60 seeds; Linux/libstdc++.")
rows = []
for f, s in FAMS:
    jb = pd.read_csv(R + f"fixed_ess_jumpbias{s}.csv", dtype={"neff": str})
    jp = pd.read_csv(R + f"fixed_ess_jumppath{s}.csv", dtype={"neff": str})
    m = jb.merge(jp, on=["copula", "estimator", "neff", "t"], suffixes=("", "_old"))
    P(f"check {f}: per-tick MSE vs fixed_ess_jumppath: {len(m)} rows, max relative difference "
      f"{(abs(m.mse - m.mse_old) / m.mse_old).max():.1e}")
    for e, _ in EST:
        for k, lab in MEM:
            g = jb[(jb.estimator == e) & (jb.neff == k) & (jb.t >= JUMP)].sort_values("t")
            b = np.abs(g.mean_err.values) / GAP
            r = recovery(b)
            pre = jb[(jb.estimator == e) & (jb.neff == k) & (jb.t >= 2000) & (jb.t < JUMP)].mean_err.mean()
            tail = np.abs(g.mean_err.values[-500:]).mean() / GAP
            rows.append(dict(copula=f, estimator=e, neff=lab, recovery=r, b_at_0=b[0], b_at_100=b[100],
                             tail_B=tail, pre_bias=pre))
df = pd.DataFrame(rows)
df.to_csv(R + "jump_recovery.csv", index=False)
P()
for f, _ in FAMS:
    P(f"=== {f}: recovery time (observations); B(100) in brackets; tail = mean B over the last 500 observations ===")
    P(f"{'N_eff':14s} " + "  ".join(f"{name:>26s}" for _, name in EST))
    for _, lab in MEM:
        cells = []
        for e, _ in EST:
            r = df[(df.copula == f) & (df.estimator == e) & (df.neff == lab)].iloc[0]
            rec = "not recovered" if pd.isna(r.recovery) else f"{int(r.recovery)}"
            cells.append(f"{rec:>13s} [{r.b_at_100:.2f}] tail {r.tail_B:.3f}")
        P(f"{lab:14s} " + "  ".join(f"{c:>26s}" for c in cells))
    P()
open(R + "jump_recovery_2026-11-12.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
