"""MFF regret (Section 6.4), Gaussian copula, from results/regret_seeds.csv (scenarios REGRET=1): cumulative MFF
log loss minus the best arm (m = 0) or the best arm sequence with at most m switches, for Hedge (delta = 0) and fixed
share (delta = 1e-4), for the mixture forecaster and the point forecast, against the Chapter 5 bound
B_{T,m}(delta) = log K + (T-1-m) log 1/(1-delta+delta/K) + m log(K/delta). Writes results/regret_2026-11-14.txt."""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

R = "results/"
K = len(C.MEMORIES)


def bound(m, d, T):
    if d == 0:
        return math.log(K) if m == 0 else float("nan")
    return math.log(K) + (T - 1 - m) * math.log(1 / (1 - d + d / K)) + m * math.log(K / d)


def ms(x):
    return f"{x.mean():7.2f} ± {2 * x.std(ddof=1) / math.sqrt(len(x)):5.2f} (med {x.median():6.2f})"


def regret_lines(g):
    out = []
    for d, dl in [(0.0, "Hedge      "), (1e-4, "fixed share")]:
        h = g[np.isclose(g.delta, d)]
        for m, c in [(0, "best0"), (1, "best1"), (2, "best2")]:
            cells = []
            for loss in ["mix", "point"]:
                x = h[f"loss_{loss}"] - h[c]
                b = np.array([bound(m, d, T) for T in h["T"]])
                v = "" if np.isnan(b).all() or loss == "point" else f" viol {int((x.values > b + 1e-9).sum())}/{len(x)} (bound {b.max():.2f})"
                cells.append(f"{loss} {ms(x)}{v}")
            out.append(f"   {dl} m={m}: " + " | ".join(cells))
    return out


def switch_line(g):
    j = g[g.delta == 0]
    sw = j[j.sw_t > 0]
    arms = ", ".join(f"{a:g}->{b:g} x{n}" for (a, b), n in sw.groupby(['sw_from', 'sw_to']).size().sort_values(ascending=False).items())
    return (f"   best one-switch sequence: {len(sw)}/{len(j)} runs switch; switch time median {sw.sw_t.median():.0f} "
            f"(IQR {sw.sw_t.quantile(.25):.0f}-{sw.sw_t.quantile(.75):.0f}; true change 2500); arms: " + arms)


def report(r):
    out = ["MFF regret (nats, cumulative over the scored ticks). Log loss, eta = 1. Gaussian copula, KLL margins, 60 seeds, Linux/libstdc++.",
           "mean ± 2 SE (median). 'viol' = runs above the Chapter 5 bound (only defined for the mixture forecaster; delta = 0 has no switching bound)."]
    for e in C.ESTIMATORS:
        out += ["", f"=== {C.EST_LABEL[e]} (T = {sorted(r[r.estimator == e]['T'].unique())}) ==="]
        for sc in C.SCENARIOS:
            g = r[(r.estimator == e) & (r.scenario == sc)]
            gain = g[g.delta == 0].best0 - g[g.delta == 0].best1
            out.append(f"-- {sc}: best fixed arm (mode) {g.arm0.mode().iloc[0]:g}; best fixed - best one-switch loss {ms(gain)}")
            out += regret_lines(g)
            a, b = (g[np.isclose(g.delta, d)].set_index("seed") for d in (1e-4, 0.0))
            out.append(f"   fixed share - Hedge (paired, = difference in cumulative MFF loss): mix {ms(a.loss_mix - b.loss_mix)} | point {ms(a.loss_point - b.loss_point)}")
            if sc == "jump":
                out.append(switch_line(g))
    return out


def main():
    lines = report(pd.read_csv(R + "regret_seeds.csv"))
    open(R + "regret_2026-11-14.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
