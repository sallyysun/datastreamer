"""How the optimal effective sample size arises and whether the bank reaches it (Gaussian drift).
(a, b) Xiao and Hermite: squared bias, variance and MSE against N_eff (results/memory_sweep.csv,
40 seeds), with the model sigma^2/N and (dN/2)^2 (sigma^2 = median var x N along the path, d = rms drift
of rho_S) and the predicted optimum N* = (2 sigma^2 / d^2)^(1/3) against the measured one.
(c) MPL log bank: its weighted memory sum_k w_k N_eff,k over time (60 seeds, results/bank_weights_paths_drift.csv)
against the local optimum N*(t) = (2 sigma^2(r_t) / d_t^2)^(1/3), with no fitted constant:
sigma^2(r) = inverse Fisher information (1 - r^2)^2 / (1 + r^2) carried to rho_S by the delta method.
See notes/studies/budget_mpl.md §20. Usage (from experiments/): python3 plot_neff_optimum.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BIAS, VAR, MSE, THEORY = "#eb6834", "#2a78d6", INK, "#52514e"
N_T, BURN = 5000, 500
t = np.arange(BURN, N_T)
import os
MATCHED = os.environ.get("SCENARIOS") == "matched"   # drift d 0.2 -> 0.8 (scenarios.hpp), else -0.5 -> 0.8
D0, DSPAN, DDEN = (0.2, 0.6, N_T - 1) if MATCHED else (-0.5, 1.3, N_T)
r_t = D0 + DSPAN * t / DDEN
drs = (3 / np.pi) / np.sqrt(1 - r_t ** 2 / 4) * DSPAN / DDEN      # d rho_S / dt
d_rms = np.sqrt(np.mean(drs ** 2))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
fig.patch.set_facecolor(SURF)


def style(a):
    a.set_facecolor(SURF)
    a.grid(True, color=GRID, lw=0.6, which="both", alpha=0.6)
    a.tick_params(labelsize=8.5, colors=MUTED)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        a.spines[s].set_color(GRID)


ms = pd.read_csv("results/memory_sweep.csv")
for a, (est, title) in zip(ax[:2], [("xiao_ew", "(a) Xiao"), ("hermite_ew", "(b) Hermite")]):
    g = ms[(ms.estimator == est) & (ms.scenario == "drift") & (ms.window == "all")].sort_values("mem_ticks")
    n = g.mem_ticks.values
    w = (n >= 50) & (n <= 2000)
    s2 = np.median((g["var"].values * n)[w])
    a.plot(n, g.bias2, "s-", color=BIAS, ms=4, lw=1.6, label="squared bias (measured)")
    a.plot(n, g["var"], "^-", color=VAR, ms=4, lw=1.6, label="variance (measured)")
    a.plot(n, g.mse, "o-", color=MSE, ms=4, lw=1.8, label="MSE (measured)")
    nn = np.logspace(np.log10(20), np.log10(30000), 200)
    a.plot(nn, s2 / nn, ":", color=VAR, lw=1.3, label=r"$\sigma^2/N$")
    a.plot(nn, (d_rms * nn / 2) ** 2, ":", color=BIAS, lw=1.3, label=r"$(dN/2)^2$")
    a.plot(nn, s2 / nn + (d_rms * nn / 2) ** 2, "--", color=THEORY, lw=1.2, label="model MSE")
    i = int(np.argmin(g.mse.values)); i = min(max(i, 1), len(n) - 2)
    x = np.log(n[i - 1:i + 2]); c2, c1, _ = np.polyfit(x, g.mse.values[i - 1:i + 2], 2)
    n_meas = np.exp(-c1 / (2 * c2))
    n_pred = (2 * s2 / d_rms ** 2) ** (1 / 3)
    a.axvline(n_meas, color=MSE, lw=1.0, alpha=0.6)
    a.axvline(n_pred, color=THEORY, lw=1.0, ls="--")
    a.text(0.03, 0.97, f"measured N* = {n_meas:.0f}\npredicted (2σ²/d²)^(1/3) = {n_pred:.0f}\nσ² = {s2:.2f}, d = {d_rms:.2e}/tick",
           transform=a.transAxes, va="top", fontsize=8.5, color=INK)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(1e-4, 0.2)
    a.set_xlabel(r"effective sample size $N_{\mathrm{eff}}$ (log)", fontsize=9, color=MUTED)
    a.set_title(f"{title}: bias–variance trade-off on drift", fontsize=10.5, color=INK, loc="left")
    style(a)
ax[0].set_ylabel(r"error$^2$ in $\rho_S$ (log)", fontsize=9, color=MUTED)
ax[0].legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)

# (c) MPL bank memory vs local optimum
b = pd.read_csv("results/bank_weights_paths_drift.csv")
b = b[(b.variant == "fixed_share") & (b.k >= 0)]
wt = b.pivot(index="t", columns="k", values="value")
ne = b.pivot(index="t", columns="k", values="neff")
mem = (wt * ne).sum(axis=1)
tt = mem.index.values
rr = D0 + DSPAN * tt / DDEN
sig2 = (3 / np.pi) ** 2 / (1 - rr ** 2 / 4) * (1 - rr ** 2) ** 2 / (1 + rr ** 2)
dd = (3 / np.pi) / np.sqrt(1 - rr ** 2 / 4) * DSPAN / DDEN
nstar = (2 * sig2 / dd ** 2) ** (1 / 3)
a = ax[2]
a.plot(tt, mem, color="#1baf7a", lw=2.0, label=r"MPL log bank: $\sum_k w_k N_{\mathrm{eff},k}$ (mean of 60 seeds)")
a.plot(tt, nstar, "--", color=INK, lw=1.4, label=r"local optimum $N^*(t)=(2\sigma^2(r_t)/d_t^2)^{1/3}$, no fitting")
for nv in [39, 99, 199, 399, 1999]:
    a.axhline(nv, color=GRID, lw=0.8)
    a.text(N_T * 1.005, nv, f"{nv}", fontsize=7.5, color=MUTED, va="center")
a.set_yscale("log")
a.set_ylim(30, 2500)
a.set_xlim(0, N_T)
a.axvspan(0, BURN, color=GRID, alpha=0.4, lw=0)
a.set_xlabel(r"tick  ($\rho_S$ drifts from " + ("0.19" if MATCHED else "−0.48") + " to 0.79)", fontsize=9, color=MUTED)
a.set_ylabel(r"effective sample size (log)", fontsize=9, color=MUTED)
a.set_title("(c) Does the bank reach the optimal memory?", fontsize=10.5, color=INK, loc="left")
a.legend(fontsize=7.8, frameon=False, loc="upper right")
style(a)
fig.suptitle("The optimal effective sample size: where it comes from, and whether the bank finds it (Gaussian drift)",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("../notes/thesis/figures/ch6_neff_optimum.png", dpi=150)
print("wrote ../notes/thesis/figures/ch6_neff_optimum.png")
w = (tt >= 1000)
print(f"bank memory vs N*(t), t>=1000: median ratio {np.median(mem.values[w] / nstar[w]):.2f}, range {np.percentile(mem.values[w]/nstar[w],10):.2f}-{np.percentile(mem.values[w]/nstar[w],90):.2f}")
