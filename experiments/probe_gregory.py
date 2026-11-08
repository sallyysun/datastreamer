"""Quick probe of Gregory (2019), "A streaming algorithm for bivariate empirical copulas" (arXiv:1805.05168):
a Greenwald-Khanna summary S1 of the first coordinate whose tuples each carry a GK summary S2^i of the
second coordinates they cover; combine every floor(1/(2 eps)) inserts (Appendix A.1-A.2); query by
C_S(u1,u2) = (n1/n) F_{n1,(2)}(F^-1_{n,(2)}(u2)) (eq. 9, Appendix A.3-A.5). Guarantee |C_S - C_emp| <= 5 eps.
Python, for a feasibility check only. See notes/studies/budget_mpl.md §22.
Usage (from experiments/): python3 probe_gregory.py -> results/gregory_probe.txt, ../notes/thesis/figures/ch6_gregory_probe.png"""
import bisect
import math
import sys
import time

import numpy as np


# ---- a GK summary as parallel lists: values v, rmin, rmax (ranks within that summary) ----
def from_tuples(v, g, d):
    rmin = np.cumsum(g).tolist()
    return {"v": list(v), "rmin": rmin, "rmax": [a + b for a, b in zip(rmin, d)]}


def to_tuples(s):
    g = [s["rmin"][0]] + [s["rmin"][i] - s["rmin"][i - 1] for i in range(1, len(s["v"]))]
    d = [s["rmax"][i] - s["rmin"][i] for i in range(len(s["v"]))]
    return list(s["v"]), g, d


def merge(a, b):
    """Greenwald & Khanna (2004) merge (Appendix A.2, eqs A.1-A.2)."""
    if not a["v"]:
        return b
    if not b["v"]:
        return a
    out = []
    for p, q in ((a, b), (b, a)):
        for k, w in enumerate(p["v"]):
            i1 = bisect.bisect_right(q["v"], w) - 1          # largest element of q <= w
            i2 = i1 + 1                                      # smallest element of q > w
            rmin = p["rmin"][k] + (q["rmin"][i1] if i1 >= 0 else 0)
            if i2 < len(q["v"]):
                rmax = p["rmax"][k] + q["rmax"][i2] - 1
            else:
                rmax = p["rmax"][k] + (q["rmax"][i1] if i1 >= 0 else 0)
            out.append((w, rmin, rmax))
    out.sort()
    return {"v": [o[0] for o in out], "rmin": [o[1] for o in out], "rmax": [o[2] for o in out]}


def compress(s, cap):
    """Merge neighbours (keeping the first and last) while g_i + g_{i+1} + Delta_{i+1} <= cap."""
    v, g, d = to_tuples(s)
    i = len(v) - 2
    while i >= 1:
        if g[i] + g[i + 1] + d[i + 1] <= cap:
            g[i + 1] += g[i]
            del v[i], g[i], d[i]
        i -= 1
    return from_tuples(v, g, d)


def quantile(s, u, eps):
    """Appendix A.3: value of approximate rank r = ceil(u n), n the summary's own count: v_L if
    r >= n - floor(eps n), else the first v_j with r_max(v_j) > r + floor(eps n)."""
    n_s = s["rmin"][-1]
    r = math.ceil(u * n_s)
    en = math.floor(eps * n_s)
    if r >= n_s - en:
        return s["v"][-1]
    for k, rm in enumerate(s["rmax"]):
        if rm > r + en:
            return s["v"][k]
    return s["v"][-1]


def inv(s, y):
    """Appendix A.4: approximate count of values <= y (r_max of the largest stored value <= y)."""
    if not s["v"] or y < s["v"][0]:
        return 0
    i = bisect.bisect_right(s["v"], y) - 1
    return s["rmax"][i] if i < len(s["v"]) - 1 else s["rmin"][-1]


class Gregory:
    def __init__(self, eps):
        self.eps, self.n = eps, 0
        self.v, self.g, self.d, self.sub = [], [], [], []      # S1 tuples and their S2 subsummaries
        self.every = max(1, int(1 / (2 * eps)))

    def insert(self, x1, x2):
        i = bisect.bisect_left(self.v, x1)
        if i == 0 or i == len(self.v):
            dd = 0
        else:
            dd = self.g[i] + self.d[i] - 1                      # Appendix A.1.1 step 3
        self.v.insert(i, x1); self.g.insert(i, 1); self.d.insert(i, dd)
        self.sub.insert(i, {"v": [x2], "rmin": [1], "rmax": [1]})
        self.n += 1
        if self.n % self.every == 0:
            self.combine()

    def combine(self):
        cap = 2 * self.eps * self.n
        j = len(self.v) - 1
        while j > 1:                                            # Appendix A.1.2
            k = 0
            tot = self.g[j] + self.d[j]
            while j - k - 1 >= 1 and tot + self.g[j - k - 1] < cap:
                k += 1
                tot += self.g[j - k]
            if k >= 1:
                q = self.sub[j - k]
                for t in range(j - k + 1, j + 1):
                    q = merge(q, self.sub[t])
                q = compress(q, 2 * self.eps * q["rmin"][-1])   # eps-approximate for its OWN count
                gsum = sum(self.g[j - k:j + 1])
                self.v[j - k:j + 1] = [self.v[j]]
                self.g[j - k:j + 1] = [gsum]
                self.d[j - k:j + 1] = [self.d[j]]
                self.sub[j - k:j + 1] = [q]
            j = j - k - 1

    def stored(self):
        return len(self.v) + sum(len(s["v"]) for s in self.sub)

    def copula_grid(self, grid):
        """C_S on grid x grid (Appendix A.5), with prefix merges P1(E) built once."""
        n = self.n
        s1 = from_tuples(self.v, self.g, self.d)
        prefix, acc = [], {"v": [], "rmin": [], "rmax": []}
        for s in self.sub:
            acc = merge(acc, s)
            prefix.append(acc)
        p2 = prefix[-1]
        out = np.zeros((len(grid), len(grid)))
        for a, u1 in enumerate(grid):
            x = quantile(s1, u1, self.eps)
            E = bisect.bisect_right(self.v, x) - 1             # eq. 8: tuples up to the returned value
            p1 = prefix[E]
            n1 = p1["rmin"][-1]
            for b, u2 in enumerate(grid):
                y = quantile(p2, u2, self.eps)
                out[a, b] = (n1 / n) * (inv(p1, y) / n1 if n1 else 0.0)
        return out


def emp_copula(x, y, grid):
    n = len(x)
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1
    out = np.zeros((len(grid), len(grid)))
    for a, u1 in enumerate(grid):
        for b, u2 in enumerate(grid):
            out[a, b] = np.mean((rx <= math.ceil(u1 * n)) & (ry <= math.ceil(u2 * n)))
    return out


def rho_from_grid(C, grid):
    h = grid[1] - grid[0]
    return 12 * C.sum() * h * h - 3


def sample(n, rho, rng, jump=None):
    r = np.full(n, rho) if jump is None else np.where(np.arange(n) < n // 2, jump[0], jump[1])
    z1 = rng.standard_normal(n)
    z2 = r * z1 + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
    return z1, z2


grid = (np.arange(20) + 0.5) / 20
lines = []
def log(s):
    print(s); lines.append(s)

# 1. accuracy vs the exact empirical copula, memory, time
log("== Gaussian copula, rho = 0.6 (rho_S 0.58), n = 5000, 3 seeds ==")
log("eps   stored(tuples)  max|C_S - C_emp|  bound 5eps  |rho_S(summary) - rho_S(sample)|  sec per 1000 inserts")
res = {}
for eps in [0.05, 0.02, 0.01]:
    errs, stor, rerr, tsec, mem_path = [], [], [], [], []
    for s in range(3):
        rng = np.random.default_rng(42 + s)
        x, y = sample(5000, 0.6, rng)
        G = Gregory(eps)
        t0 = time.time()
        path = []
        for i in range(5000):
            G.insert(x[i], y[i])
            if (i + 1) % 250 == 0:
                path.append(G.stored())
        tsec.append((time.time() - t0) / 5)
        Cs, Ce = G.copula_grid(grid), emp_copula(x, y, grid)
        errs.append(np.abs(Cs - Ce).max())
        stor.append(G.stored())
        rerr.append(abs(rho_from_grid(Cs, grid) - rho_from_grid(Ce, grid)))
        mem_path.append(path)
    res[eps] = (np.mean(mem_path, axis=0), np.mean(errs))
    log(f"{eps:<5} {np.mean(stor):>14.0f}  {np.mean(errs):>16.4f}  {5 * eps:>10.3f}  {np.mean(rerr):>31.4f}  {np.mean(tsec):>8.3f}")

# 2. memory growth with n
log("\n== memory (stored tuples, S1 + all S2) against n, eps = 0.02, one seed ==")
rng = np.random.default_rng(7)
x, y = sample(20000, 0.6, rng)
G = Gregory(0.02)
grow = []
for i in range(20000):
    G.insert(x[i], y[i])
    if (i + 1) in (1000, 2000, 5000, 10000, 20000):
        grow.append((i + 1, G.stored()))
        log(f"n = {i + 1:>6}: {G.stored()} tuples ({3 * G.stored() * 8 / 1024:.1f} KiB as 3 doubles each)")

# 3. a jump: the summary is cumulative
log("\n== jump rho 0.2 -> 0.8 at n/2 (rho_S 0.19 -> 0.79), eps = 0.02: rho_S read from the summary ==")
rng = np.random.default_rng(11)
x, y = sample(5000, None, rng, jump=(0.2, 0.8))
G = Gregory(0.02)
jpath = []
for i in range(5000):
    G.insert(x[i], y[i])
    if (i + 1) % 250 == 0:
        jpath.append((i + 1, rho_from_grid(G.copula_grid(grid), grid)))
for t, r in jpath[3::4]:
    log(f"t = {t:>5}: rho_S(summary) = {r:.3f}   truth = {0.19 if t <= 2500 else 0.79:.2f}")

open("results/gregory_probe.txt", "w").write("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
fig.patch.set_facecolor(SURF)
cols = {0.05: "#9ec5f4", 0.02: "#3987e5", 0.01: "#103d73"}
for eps, (path, err) in res.items():
    ax[0].plot(np.arange(250, 5001, 250), path, color=cols[eps], lw=1.8, label=f"ε = {eps} (max error {err:.3f}, bound {5 * eps:.2f})")
ax[0].set_title("(a) Stored tuples while the stream grows", fontsize=10, color=INK, loc="left")
ax[0].set_xlabel("n", fontsize=9, color=MUTED); ax[0].set_ylabel("tuples in S1 + all S2", fontsize=9, color=MUTED)
ax[0].legend(fontsize=7.8, frameon=False)
nn, ss = zip(*grow)
ax[1].plot(nn, ss, "o-", color="#3987e5", lw=1.8)
ax[1].set_xscale("log")
ax[1].set_title("(b) Memory against n (ε = 0.02): grows with log n", fontsize=10, color=INK, loc="left")
ax[1].set_xlabel("n (log)", fontsize=9, color=MUTED); ax[1].set_ylabel("stored tuples", fontsize=9, color=MUTED)
tt, rr = zip(*jpath)
ax[2].plot(tt, rr, color="#3987e5", lw=1.8, label="ρ_S from the Gregory summary")
ax[2].plot([0, 2500, 2500, 5000], [0.19, 0.19, 0.79, 0.79], "--", color=INK, lw=1.2, label="true ρ_S")
ax[2].axvline(2500, color=MUTED, ls=":", lw=1)
ax[2].set_title("(c) A jump: the summary is cumulative, no forgetting", fontsize=10, color=INK, loc="left")
ax[2].set_xlabel("n", fontsize=9, color=MUTED); ax[2].legend(fontsize=7.8, frameon=False, loc="upper left")
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6); a.tick_params(labelsize=8.5, colors=MUTED)
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)
fig.suptitle("Gregory (2019) streaming empirical-copula summary: quick probe (Gaussian copula, Python)", fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("../notes/thesis/figures/ch6_gregory_probe.png", dpi=150)
print("wrote figure and results/gregory_probe.txt")
