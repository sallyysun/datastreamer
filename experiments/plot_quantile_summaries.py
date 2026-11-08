"""Illustration: how Greenwald-Khanna (GK) and KLL quantile summaries store a stream, on 3000 N(0,1)
values. Simplified textbook versions in Python, for the figure only; the experiments use KllSketch /
Apache DataSketches (src/estimators/dependence/). See notes/studies/budget_mpl.md §21.
Usage (from experiments/): python3 plot_quantile_summaries.py -> ../notes/thesis/figures/ch4_quantile_summaries.png"""
import bisect
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class GK:
    """Greenwald & Khanna (2001): sorted tuples (v, g, delta). rmin(v_i) = sum_{j<=i} g_j,
    rmax(v_i) = rmin + delta_i; every rank query is within eps*n. Compress merges neighbours whose
    combined uncertainty stays within 2*eps*n."""
    def __init__(self, eps):
        self.eps, self.n, self.t = eps, 0, []   # t: list of [v, g, delta]

    def insert(self, x):
        vs = [a[0] for a in self.t]
        i = bisect.bisect_left(vs, x)
        d = 0 if i == 0 or i == len(self.t) else int(2 * self.eps * self.n)
        self.t.insert(i, [x, 1, d])
        self.n += 1
        if self.n % max(1, int(1 / (2 * self.eps))) == 0:
            self.compress()

    def compress(self):
        cap = 2 * self.eps * self.n
        i = len(self.t) - 2
        while i >= 1:
            a, b = self.t[i], self.t[i + 1]
            if a[1] + b[1] + b[2] <= cap:
                b[1] += a[1]
                del self.t[i]
            i -= 1

    def bounds(self):
        rmin = np.cumsum([a[1] for a in self.t])
        rmax = rmin + np.array([a[2] for a in self.t])
        return np.array([a[0] for a in self.t]), rmin, rmax

    def cdf(self, x):
        # x falls between stored values v_{i-1} < x <= v_i: the number of observations below x lies in
        # [rmin_{i-1}, rmax_i - 1], an interval of width g_i + delta_i - 1 <= 2 eps n; report its midpoint.
        v, rmin, rmax = self.bounds()
        i = bisect.bisect_left(list(v), x)
        lo = rmin[i - 1] if i > 0 else 0
        hi = rmax[i] - 1 if i < len(v) else self.n
        return 0.5 * (lo + hi) / self.n


class KLL:
    """Karnin, Lang & Liberty (2016): levels h = 0, 1, ...; an item at level h stands for 2^h
    observations. When the levels overflow, the lowest full level is sorted and every other item
    (random offset) is promoted; capacities shrink geometrically, k (2/3)^depth."""
    def __init__(self, k, seed=1):
        self.k, self.levels, self.n, self.rng = k, [[]], 0, random.Random(seed)

    def cap(self, h):
        return max(2, int(np.ceil(self.k * (2 / 3) ** (len(self.levels) - 1 - h))))

    def insert(self, x):
        self.levels[0].append(x)
        self.n += 1
        if sum(len(l) for l in self.levels) > sum(self.cap(h) for h in range(len(self.levels))):
            for h in range(len(self.levels)):
                if len(self.levels[h]) >= self.cap(h):
                    if h + 1 == len(self.levels):
                        self.levels.append([])
                    lv = sorted(self.levels[h])
                    keep = [lv.pop()] if len(lv) % 2 else []
                    off = self.rng.randint(0, 1)
                    self.levels[h + 1] += lv[off::2]
                    self.levels[h] = keep
                    break

    def cdf(self, x):
        return sum((2 ** h) * sum(v < x for v in l) for h, l in enumerate(self.levels)) / self.n


rng = np.random.default_rng(3)
data = rng.standard_normal(3000)
gk, kll = GK(eps=0.02), KLL(k=40)
for x in data:
    gk.insert(x)
    kll.insert(x)
srt = np.sort(data)
n = len(data)
grid = np.linspace(-3, 3, 400)
true = np.searchsorted(srt, grid) / n

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GKC, KLLC = "#6250d6", "#1baf7a"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
fig.patch.set_facecolor(SURF)

# (a) GK
v, rmin, rmax = gk.bounds()
a = ax[0]
a.plot(srt, np.arange(1, n + 1) / n, color=GRID, lw=3, label="exact empirical CDF")
a.vlines(v, rmin / n, rmax / n, color=GKC, lw=2.2)
a.plot(v, rmin / n, "_", color=GKC, ms=7)
a.plot(v, rmax / n, "_", color=GKC, ms=7, label=r"stored value with rank interval [$r_{\min}$, $r_{\max}$]/n")
a.set_title(f"(a) Greenwald–Khanna: {len(v)} stored tuples (v, g, Δ), ε = {gk.eps}\n"
            f"each value keeps its count g and rank uncertainty Δ; every rank within εn = {gk.eps * n:.0f}",
            fontsize=9.5, color=INK, loc="left")
a.set_xlabel("value", fontsize=9, color=MUTED); a.set_ylabel("rank / n", fontsize=9, color=MUTED)
a.legend(fontsize=7.8, frameon=False, loc="lower right")

# (b) KLL levels
a = ax[1]
H = len(kll.levels)
for h, lv in enumerate(kll.levels):
    a.plot(lv, np.full(len(lv), h), "o", color=KLLC, ms=4 + 1.2 * h, alpha=0.85, mec=SURF, mew=0.5)
    a.text(3.7, h, f"weight 2^{h} = {2 ** h}   ({len(lv)} stored)", fontsize=7.8, color=MUTED, va="center")
a.set_yticks(range(H)); a.set_yticklabels([f"level {h}" for h in range(H)], fontsize=8.5)
a.set_xlim(-3.2, 8.2)
a.set_ylim(-0.6, H - 0.4)
a.set_title(f"(b) KLL: {sum(len(l) for l in kll.levels)} stored values in {H} levels, k = {kll.k}\n"
            "a full level is sorted; every other value moves up, doubling its weight",
            fontsize=9.5, color=INK, loc="left")
a.set_xlabel("value", fontsize=9, color=MUTED)

# (c) rank error
a = ax[2]
a.plot(grid, [gk.cdf(x) for x in grid] - true, color=GKC, lw=1.6, label=f"GK ({len(v)} tuples × 3 numbers)")
a.plot(grid, [kll.cdf(x) for x in grid] - true, color=KLLC, lw=1.6, label=f"KLL ({sum(len(l) for l in kll.levels)} values)")
a.axhspan(-gk.eps, gk.eps, color=GKC, alpha=0.08, lw=0, label="GK guarantee ±ε")
a.axhline(0, color=MUTED, lw=0.8)
a.set_title("(c) Rank error, about equal memory\n(≈ 114 stored numbers each)", fontsize=9.5, color=INK, loc="left")
a.set_xlabel("value", fontsize=9, color=MUTED); a.set_ylabel(r"$\hat F(x) - F_n(x)$", fontsize=9, color=MUTED)
a.legend(fontsize=7.8, frameon=False, loc="lower left")
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6, alpha=0.7); a.tick_params(labelsize=8.5, colors=MUTED)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        a.spines[s].set_color(GRID)
fig.suptitle(f"How the two quantile summaries store a stream of n = {n} values (N(0,1)), and how accurately they rank",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("../notes/thesis/figures/ch4_quantile_summaries.png", dpi=150)
gke = np.abs(np.array([gk.cdf(x) for x in grid]) - true).max(); kle = np.abs(np.array([kll.cdf(x) for x in grid]) - true).max()
print(f"GK {len(v)} tuples, max rank error {gke:.4f}; KLL {sum(len(l) for l in kll.levels)} items, {H} levels, max rank error {kle:.4f}")
