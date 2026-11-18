"""Compare real-data MFF and VFF with the best fixed-arm switching paths."""
import itertools
import sys

import numpy as np
import pandas as pd

from real_common import FIX, eval_days, outputs, read, result, write_report

MS = [0, 1, 2, 5]
EST = {"RMPL": "practical_rmpl", "Counting matrix": "practical_xiao", "Hermite": "practical_hermite"}
SET = {"t4": ("lt_", "lt_mffmix_t", "lt_mff_t"), "Gaussian": ("lg_", "lg_mffmix_g", "lg_mff_g")}


def best_seq(Lk, m):
    """Lk: (K, T) losses. Returns (min cost over <= m switches, path of arm indices)."""
    K, T = Lk.shape
    D = np.full((m + 1, K), np.inf)
    D[0] = Lk[:, 0]
    back = np.zeros((T, m + 1, K, 2), dtype=np.int16)
    arms = np.arange(K)
    for t in range(1, T):
        E = np.full((m + 1, K), np.inf)
        for j in range(m + 1):
            stay = D[j]
            if j > 0:
                prev = D[j - 1]
                order = np.argsort(prev)[:2]
                sw = np.where(arms == order[0], prev[order[1]], prev[order[0]])   # best other arm
                fromk = np.where(arms == order[0], order[1], order[0])
            else:
                sw, fromk = np.full(K, np.inf), np.zeros(K, int)
            take_sw = sw < stay
            E[j] = Lk[:, t] + np.where(take_sw, sw, stay)
            back[t, j, :, 0] = np.where(take_sw, j - 1, j)
            back[t, j, :, 1] = np.where(take_sw, fromk, arms)
        D = E
    j, k = np.unravel_index(np.argmin(D), D.shape)
    cost, path = D[j, k], np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = k
        if t > 0:
            j, k = back[t, j, k]
    return cost, path


def brute_force_check():
    rng = np.random.default_rng(1)
    for _ in range(200):
        T = 7
        Lk = rng.normal(size=(3, T))
        m = rng.integers(0, 3)
        best = min(sum(Lk[s[t], t] for t in range(T)) for s in itertools.product(range(3), repeat=T)
                   if sum(s[t] != s[t - 1] for t in range(1, T)) <= m)
        c, p = best_seq(Lk, m)
        assert abs(c - best) < 1e-9 and abs(Lk[p, np.arange(T)].sum() - c) < 1e-9 and (np.diff(p) != 0).sum() <= m
    print("CHECK brute force (200 random short series): DP optimum, traced path cost and switch budget all agree")


def regrets(S):
    rows, paths = [], []
    for est, d in EST.items():
        for pair, f in outputs(S, d).items():
            x = eval_days(read(f)).reset_index(drop=True)
            T = len(x)
            for fam, (pre, mix, plug) in SET.items():
                Lk = -x[[f"{pre}fixed_{k}" for k in FIX]].values.T
                static = Lk.sum(1).min()
                for m in MS:
                    cost, path = best_seq(Lk, m)
                    assert (np.diff(path) != 0).sum() <= m and abs(Lk[path, np.arange(T)].sum() - cost) < 1e-6
                    assert m > 0 or abs(cost - static) < 1e-6
                    for meth, c in (("MFF", mix), ("VFF", f"{pre}vff"), ("MFF single density", plug)):
                        rows.append(dict(estimator=est, copula=fam, m=m, method=meth, pair=pair, regret=1e3 * (-x[c].values.sum() - cost) / T))
                    if m in (1, 2):
                        sw = np.flatnonzero(np.diff(path) != 0) + 1
                        paths.append(dict(estimator=est, copula=fam, m=m, pair=pair, switches=len(sw),
                                          switch_dates=";".join(str(x.date.iloc[i].date()) for i in sw),
                                          segments="->".join(str(FIX[s[0]]) for s in np.split(path, sw)),
                                          gain_vs_static=1e3 * (static - cost) / T))
    print("CHECK every pair/estimator/copula: m = 0 equals the static oracle; traced paths respect the budget and reproduce the cost")
    return pd.DataFrame(rows), pd.DataFrame(paths)


def report(Rg, Pt):
    out = ["Switching regret vs the best sequence of fixed-ESS arms with <= m switches, x1e-3 nats/day/pair (mean over 32 pairs; lower is better)"]
    for fam in ("t4", "Gaussian"):
        out.append(f"\n=== {fam} copula      " + "".join(f"   m={m:<5d}" for m in MS) + "   pairs with regret < 0 at m=1")
        for est in EST:
            for meth in ("MFF", "VFF", "MFF single density"):
                g = Rg[(Rg.estimator == est) & (Rg.copula == fam) & (Rg.method == meth)]
                neg = int((g[g.m == 1].regret < 0).sum())
                out.append(f"  {est:16s} {meth:19s}" + "".join(f" {g[g.m == m].regret.mean():+8.2f}" for m in MS) + f"   {neg}/32")
    out.append("\nOracle sequences (t4 copula): how much switching buys over the static oracle, and where the switches fall")
    for est in EST:
        for m in (1, 2):
            g = Pt[(Pt.estimator == est) & (Pt.copula == "t4") & (Pt.m == m)]
            out.append(f"  {est:16s} m={m}: pairs using all {m} switch(es) {int((g.switches == m).sum())}/32; oracle gain over static "
                       f"{g.gain_vs_static.mean():.2f} (median {g.gain_vs_static.median():.2f}); most common sequences {g.segments.value_counts().head(4).to_dict()}")
        g = Pt[(Pt.estimator == est) & (Pt.copula == "t4") & (Pt.m == 1) & (Pt.switches == 1)]
        out.append(f"  {'':16s} m=1 switch years: {pd.to_datetime(g.switch_dates).dt.year.value_counts().sort_index().to_dict()}")
    return out


def main():
    brute_force_check()
    Rg, Pt = regrets(sys.argv[1])
    Rg.to_csv(result("switching_per_pair.csv"), index=False)
    Pt.to_csv(result("switching_paths.csv"), index=False)
    out = report(Rg, Pt)
    write_report(result("switching_results.txt"), out)


if __name__ == "__main__":
    main()
