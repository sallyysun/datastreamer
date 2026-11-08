# Gregory probe, part 2: the paper's r_max-based queries (A.3, A.4) against midpoint queries. Reuses probe_gregory.py. See budget_mpl.md §22.
import sys, bisect, math, numpy as np
src=open("probe_gregory.py").read().split("grid = (np.arange(20)")[0]
exec(src)
def quantile_mid(s, u, eps):
    n_s = s["rmin"][-1]; r = u * n_s
    mids = [(a + b) / 2 for a, b in zip(s["rmin"], s["rmax"])]
    k = int(np.argmin([abs(m - r) for m in mids])); return s["v"][k]
def inv_mid(s, y):
    if not s["v"] or y < s["v"][0]: return 0
    i = bisect.bisect_right(s["v"], y) - 1
    if i >= len(s["v"]) - 1: return s["rmin"][-1]
    return 0.5 * (s["rmin"][i] + s["rmax"][i + 1] - 1)
grid=(np.arange(20)+0.5)/20
quantile_paper, inv_paper = quantile, inv
for eps in [0.02, 0.01]:
    out={"paper":[], "midpoint":[]}; rr={"paper":[], "midpoint":[]}
    for s in range(3):
        rng=np.random.default_rng(42+s); x,y=sample(5000,0.6,rng); G=Gregory(eps)
        for i in range(5000): G.insert(x[i],y[i])
        Ce=emp_copula(x,y,grid)
        for name,(q,iv) in {"paper":(quantile,inv),"midpoint":(quantile_mid,inv_mid)}.items():
            globals()["quantile"],globals()["inv"]=q,iv
            Cs=G.copula_grid(grid)
            out[name].append(np.abs(Cs-Ce).max()); rr[name].append(rho_from_grid(Cs,grid)-rho_from_grid(Ce,grid))
        globals()["quantile"],globals()["inv"]=out and quantile_paper, inv_paper
    for name in out: print(f"eps={eps} {name:9s}: max|C_S-C_emp| {np.mean(out[name]):.4f}  rho_S error {np.mean(rr[name]):+.3f}")
