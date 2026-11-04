"""Coverage of online 90% intervals for the EW corner counter chi_hat(0.05), smoothed target, 60 seeds:
(a) pointwise (whole run) for the analytic interval, the iid-multiplier bootstrap and the AR-multiplier bootstrap
of Nagler, Brock and Palm (2026); (b) uniform-in-time coverage of the two bootstrap bands over (t1, T].
Reads results/tail_ci[_uniform]_<regime>_42.csv (probe_tail_ci).
Usage (from experiments/): python3 plot_tail_ci.py -> ../notes/thesis/figures/ch6_tail_ci.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"analytic": "#eb6834", "iid": "#1baf7a", "ar": "#2a78d6"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
regs = [("gaussian", "Gaussian,\nindependent"), ("t_persist", "$t_4$, clustered\nextremes"),
        ("gauss_ar", "Gaussian,\nAR(1) φ = 0.7")]
mems = [399, 1999, 4999]
d = pd.concat(pd.read_csv(f"results/tail_ci_{r}_42.csv") for r, _ in regs)
u = pd.concat(pd.read_csv(f"results/tail_ci_uniform_{r}_42.csv") for r, _ in regs)
d = d[(d.q == 0.05) & (d.target == "smoothed") & (d.window == "whole")]
u = u[(u.q == 0.05) & (u.target == "smoothed") & (u.horizon == "t1_T")]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
x0 = np.arange(len(regs) * len(mems))
for k, (iv, lab, key) in enumerate([("analytic", "analytic (independence)", "analytic"),
                                    ("iid_point", "bootstrap, iid multipliers", "iid"),
                                    ("ar_point", "bootstrap, AR multipliers", "ar")]):
    y = [d[(d.regime == r) & (d.M == m) & (d.interval == iv)].coverage.mean() for r, _ in regs for m in mems]
    ax[0].plot(x0 + (k - 1) * 0.22, y, "o", color=COL[key], ms=5, label=lab)
for k, (iv, lab, key) in enumerate([("iid_band", "band, iid multipliers", "iid"), ("ar_band", "band, AR multipliers", "ar")]):
    y = [u[(u.regime == r) & (u.M == m) & (u.interval == iv)].covered.mean() for r, _ in regs for m in mems]
    ax[1].plot(x0 + (k - 0.5) * 0.25, y, "o", color=COL[key], ms=5, label=lab)
for a, title, yl in [(ax[0], "(a) Pointwise coverage, whole run", (0.75, 1.0)),
                     (ax[1], "(b) Uniform-in-time coverage over ticks 1000–10000", (0.0, 1.0))]:
    a.axhline(0.9, color=INK, lw=0.8, ls="--")
    a.set_xticks(x0, [f"M={m}" for _ in regs for m in mems], fontsize=7)
    for i, (_, rl) in enumerate(regs):
        a.text(i * 3 + 1, yl[0] - (yl[1] - yl[0]) * 0.2, rl, ha="center", va="top", fontsize=8, color=INK)
        if i: a.axvline(i * 3 - 0.5, color=GRID, lw=0.8)
    a.set_ylim(*yl)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_title(title, loc="left", fontsize=9, color=INK)
    a.set_ylabel("coverage of nominal 90%")
fig.suptitle(r"Online 90% intervals for the EW corner counter $\hat\chi_L(0.05)$, target $\bar\chi_t$, 60 seeds",
             fontsize=9, color=INK)
for a in ax:
    a.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig("../notes/thesis/figures/ch6_tail_ci.png", dpi=170)
print("wrote ch6_tail_ci.png")
