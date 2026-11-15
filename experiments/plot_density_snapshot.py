"""Exploratory: predicted Gaussian copula densities after the abrupt jump (RMPL, Gaussian copula, ONE predetermined seed: 42,
the first evaluation seed). Forecasts from results/strategies_dump_jump_s42.csv (probe_strategies DUMP_SEED=42; matches
strategies_seeds for that seed to 5e-9). Each forecast rho_S -> Gaussian copula with r = 2 sin(pi rho / 6), the
transformation used by the density evaluation (copula::param_from_spearman). Fixed = Gaussian Hellinger-best N_eff for
RMPL on the jump path (N_eff 99). Times: n = 2525 (25 observations after the jump) and n = 2650 (150 after).
H^2 in each panel: closed form for two Gaussian copulas, 1 - (det S1 det S2)^(1/4) / det((S1 + S2)/2)^(1/2).
Usage (repo root): python3 experiments/plot_density_snapshot.py
  -> images/ch6/explore_density_snapshot.png (log densities), explore_density_snapshot_diff.png (log ratios)"""
import sys
import numpy as np
import pandas as pd
from statistics import NormalDist

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
D = pd.read_csv(core.R + "strategies_dump_jump_s42.csv")
D = D[D.estimator == "rmpl"]
TIMES = [(2525, "25 after the jump"), (2650, "150 after the jump")]
STRATS = [("fixed_99", "Fixed ($N_{\\mathrm{eff}}=99$)"), ("vff_slow", "VFF ($\\alpha=0.02$)"), ("mff", "MFF")]
u = np.linspace(0.0025, 0.9975, 240)
U, V = np.meshgrid(u, u)
_ppf = np.vectorize(NormalDist().inv_cdf)
X, Y = _ppf(U), _ppf(V)


def r_of(rho):
    return 2 * np.sin(np.pi * rho / 6)


def logc(r):
    return -0.5 * np.log(1 - r * r) - (r * r * (X ** 2 + Y ** 2) - 2 * r * X * Y) / (2 * (1 - r * r))


def h2(r1, r2):
    return 1 - ((1 - r1 * r1) * (1 - r2 * r2)) ** 0.25 / np.sqrt(1 - ((r1 + r2) / 2) ** 2)


def get(strategy, t):
    row = D[(D.strategy == strategy) & (D.t == t)].iloc[0]
    return row.forecast, row.rho_true


cells = {}
for t, _ in TIMES:
    rho_true = get("mff", t)[1]
    cells[(t, "truth")] = (rho_true, r_of(rho_true))
    for k, _ in STRATS:
        rho = get(k, t)[0]
        cells[(t, k)] = (rho, r_of(rho))
for (t, k), (rho, r) in cells.items():
    print(f"n={t} {k:9s} rho_S={rho:.3f} r={r:.3f}" + ("" if k == "truth" else f"  H2 vs truth={h2(r, cells[(t, 'truth')][1]):.4f}"))

# 1) log densities, shared colour scale
L = {key: logc(r) for key, (rho, r) in cells.items()}
vmin, vmax = -3.0, max(v.max() for v in L.values())
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(2, 4, figsize=(core.W, 3.4), sharex=True, sharey=True)
    for ri, (t, tl) in enumerate(TIMES):
        cols = [("truth", r"True ($\rho_S$=%.2f)" % cells[(t, "truth")][0])] + [(k, lab) for k, lab in STRATS]
        for ci, (k, lab) in enumerate(cols):
            ax = axs[ri, ci]
            im = ax.imshow(np.clip(L[(t, k)], vmin, None), origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=vmin, vmax=vmax)
            rho, r = cells[(t, k)]
            sub = "" if k == "truth" else "\n" + r"$\hat\rho_S$=%.2f, $H^2$=%.3f" % (rho, h2(r, cells[(t, 'truth')][1]))
            ax.set_title((lab if ri == 0 else lab.split(" (")[0] if k != "truth" else "True") + sub, fontsize=7)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        axs[ri, 0].set_ylabel(f"n = {t}\n(+{t - 2500})\nv", fontsize=7.5)
    for ax in axs[1]: ax.set_xlabel("u")
    fig.subplots_adjust(left=0.13, right=0.88, top=0.86, bottom=0.12, hspace=0.6, wspace=0.12)
    cb = fig.colorbar(im, cax=fig.add_axes([0.9, 0.2, 0.015, 0.6]))
    cb.set_label("log density (below -3 shown at -3)", fontsize=7)
    fig.savefig("images/ch6/explore_density_snapshot.png"); plt.close(fig)

# 2) log ratio log c_hat - log c_true, shared symmetric scale
R = {(t, k): L[(t, k)] - L[(t, "truth")] for t, _ in TIMES for k, _ in STRATS}
lim = 3.0
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(2, 3, figsize=(core.W * 0.82, 3.6), sharex=True, sharey=True)
    for ri, (t, tl) in enumerate(TIMES):
        for ci, (k, lab) in enumerate(STRATS):
            ax = axs[ri, ci]
            im = ax.imshow(np.clip(R[(t, k)], -lim, lim), origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-lim, vmax=lim)
            rho, r = cells[(t, k)]
            ax.set_title((lab if ri == 0 else lab.split(" (")[0]) + "\n" + r"$\hat\rho_S$=%.2f, $H^2$=%.3f" % (rho, h2(r, cells[(t, 'truth')][1])), fontsize=7)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        axs[ri, 0].set_ylabel(f"n = {t} ({tl})\n" + r"true $\rho_S$=%.2f" % cells[(t, "truth")][0] + "\nv", fontsize=7.5)
    for ax in axs[1]: ax.set_xlabel("u")
    fig.subplots_adjust(left=0.18, right=0.85, top=0.86, bottom=0.12, hspace=0.6, wspace=0.12)
    cb = fig.colorbar(im, cax=fig.add_axes([0.875, 0.2, 0.018, 0.6]))
    cb.set_label(r"$\log\hat c - \log c$ (colour clipped at $\pm3$)", fontsize=7)
    fig.savefig("images/ch6/explore_density_snapshot_diff.png"); plt.close(fig)
print("wrote images/ch6/explore_density_snapshot.png, explore_density_snapshot_diff.png")
