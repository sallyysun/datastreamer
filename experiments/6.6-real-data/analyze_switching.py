"""Switching regret of MFF and VFF (practical application). For each pair, estimator (RMPL / counting matrix / Hermite)
and copula (t4 primary, Gaussian check), the comparator is the best sequence of that estimator's five fixed-ESS arms with
at most m switches over the pair's evaluation days (t >= 560, 2008 onward), found by dynamic programming with traceback.
Regret per observation = (sum l^A - min_{<=m switches} sum l^(k_n)) / N, averaged over the 32 pairs, x1e-3 nats/day/pair.
m in {0, 1, 2, 5}. A = MFF (mixture), VFF, MFF (single density). Point estimates only (the oracle sequence is chosen
from the evaluation data); the paired MFF - VFF loss difference, in which the oracle cancels, carries the inference.
Checks: m = 0 equals the static oracle; traced paths respect the budget and reproduce the DP cost; brute force on short
series. Usage: python3 analyze_switching.py <scratch dir>"""
import sys, glob, os, itertools, numpy as np, pandas as pd
S = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
FIX = [39, 99, 199, 399, 1999]; MS = [0, 1, 2, 5]
def best_seq(Lk, m):
    """Lk: (K, T) losses. Returns (min cost over <= m switches, path of arm indices)."""
    K, T = Lk.shape; inf = np.inf
    D = np.full((m + 1, K), inf); D[0] = Lk[:, 0]; back = np.zeros((T, m + 1, K, 2), dtype=np.int16)
    for t in range(1, T):
        E = np.full((m + 1, K), inf)
        for j in range(m + 1):
            stay = D[j]                                          # same arm, same switch count
            if j > 0:
                prev = D[j - 1]; order = np.argsort(prev)[:2]
                sw = np.where(np.arange(K) == order[0], prev[order[1]], prev[order[0]])   # best other arm
                fromk = np.where(np.arange(K) == order[0], order[1], order[0])
            else:
                sw = np.full(K, inf); fromk = np.zeros(K, int)
            take_sw = sw < stay
            E[j] = Lk[:, t] + np.where(take_sw, sw, stay)
            back[t, j, :, 0] = np.where(take_sw, j - 1, j); back[t, j, :, 1] = np.where(take_sw, fromk, np.arange(K))
        D = E
    j, k = np.unravel_index(np.argmin(D), D.shape); cost = D[j, k]; path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = k
        if t > 0: j, k = back[t, j, k]
    return cost, path
# brute-force check on short random series
rng = np.random.default_rng(1)
for _ in range(200):
    T = 7; Lk = rng.normal(size=(3, T)); m = rng.integers(0, 3)
    best = min(sum(Lk[s[t], t] for t in range(T)) for s in itertools.product(range(3), repeat=T) if sum(s[t] != s[t - 1] for t in range(1, T)) <= m)
    c, p = best_seq(Lk, m); assert abs(c - best) < 1e-9 and abs(Lk[p, np.arange(T)].sum() - c) < 1e-9 and (np.diff(p) != 0).sum() <= m
print("CHECK brute force (200 random short series): DP optimum, traced path cost and switch budget all agree")
EST = {"RMPL": "practical_rmpl", "Counting matrix": "practical_xiao", "Hermite": "practical_hermite"}
SET = {"t4": ("lt_", "lt_mffmix_t", "lt_mff_t"), "Gaussian": ("lg_", "lg_mffmix_g", "lg_mff_g")}
rows, paths, out = [], [], []
def pr(s=""): print(s); out.append(s)
for est, d in EST.items():
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        x = pd.read_csv(f, parse_dates=["date"]).iloc[560:]; x = x[x.date >= "2008-01-01"].reset_index(drop=True)
        pair = os.path.basename(f)[:-4]; T = len(x)
        for fam, (pre, mix, plug) in SET.items():
            Lk = -x[[f"{pre}fixed_{k}" for k in FIX]].values.T
            static = Lk.sum(1).min()
            for m in MS:
                cost, path = best_seq(Lk, m)
                assert (np.diff(path) != 0).sum() <= m and abs(Lk[path, np.arange(T)].sum() - cost) < 1e-6
                if m == 0: assert abs(cost - static) < 1e-6
                for meth, c in (("MFF", mix), ("VFF", f"{pre}vff"), ("MFF single density", plug)):
                    rows.append(dict(estimator=est, copula=fam, m=m, method=meth, pair=pair, regret=1e3 * (-x[c].values.sum() - cost) / T))
                if m in (1, 2):
                    sw = np.flatnonzero(np.diff(path) != 0) + 1; segs = np.split(path, sw)
                    paths.append(dict(estimator=est, copula=fam, m=m, pair=pair, switches=len(sw),
                                      switch_dates=";".join(str(x.date.iloc[i].date()) for i in sw), segments="->".join(str(FIX[s[0]]) for s in segs),
                                      gain_vs_static=1e3 * (static - cost) / T))
print("CHECK every pair/estimator/copula: m = 0 equals the static oracle; traced paths respect the budget and reproduce the cost")
Rg = pd.DataFrame(rows); Pt = pd.DataFrame(paths)
Rg.to_csv(os.path.join(HERE, "results", "switching_per_pair.csv"), index=False); Pt.to_csv(os.path.join(HERE, "results", "switching_paths.csv"), index=False)
pr("Switching regret vs the best sequence of fixed-ESS arms with <= m switches, x1e-3 nats/day/pair (mean over 32 pairs; lower is better)")
for fam in ("t4", "Gaussian"):
    pr(f"\n=== {fam} copula      " + "".join(f"   m={m:<5d}" for m in MS) + "   pairs with regret < 0 at m=1")
    for est in EST:
        for meth in ("MFF", "VFF", "MFF single density"):
            g = Rg[(Rg.estimator == est) & (Rg.copula == fam) & (Rg.method == meth)]
            vals = [g[g.m == m].regret.mean() for m in MS]; neg = int((g[g.m == 1].regret < 0).sum())
            pr(f"  {est:16s} {meth:19s}" + "".join(f" {v:+8.2f}" for v in vals) + f"   {neg}/32")
pr("\nOracle sequences (t4 copula): how much switching buys over the static oracle, and where the switches fall")
for est in EST:
    for m in (1, 2):
        g = Pt[(Pt.estimator == est) & (Pt.copula == "t4") & (Pt.m == m)]
        pr(f"  {est:16s} m={m}: pairs using all {m} switch(es) {int((g.switches == m).sum())}/32; oracle gain over static "
           f"{g.gain_vs_static.mean():.2f} (median {g.gain_vs_static.median():.2f}); most common sequences {g.segments.value_counts().head(4).to_dict()}")
    g = Pt[(Pt.estimator == est) & (Pt.copula == "t4") & (Pt.m == 1) & (Pt.switches == 1)]
    yrs = pd.to_datetime(g.switch_dates).dt.year.value_counts().sort_index().to_dict()
    pr(f"  {'':16s} m=1 switch years: {yrs}")
open(os.path.join(HERE, "results", "switching_results.txt"), "w").write("\n".join(out) + "\n")
