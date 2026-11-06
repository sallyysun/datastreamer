# Copula ISE in absolute terms, and against rho_S RMSE, from probe_batch_ise output (no new runs).
# See notes/studies/thesis_bank_ise_batch.md. Usage (repo root): python3 experiments/plot_ise_levels.py
import glob
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
M = {"family_bank": ("Family bank (24 parametric arms)", "#2a78d6", "o"),
     "mle_bank_log": ("MPL bank, Gaussian log loss", "#eb6834", "s"),
     "xiao_bank_log": ("Xiao bank, Gaussian log loss", "#1baf7a", "^"),
     "xiao_bank_own": ("Xiao bank, own density", "#eda100", "D"),
     "hermite_bank_own": ("Hermite bank, own density", "#e87ba4", "v")}

d = pd.concat(pd.read_csv(f) for f in glob.glob("experiments/results/batch_ise_42*.csv"))
d = d[(d.window == "whole") & d.method.isin(M)]
g = d.groupby(["copula", "scenario", "method", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
g["se"] = g["std"] / np.sqrt(g["count"])
w = g.pivot_table(index=["copula", "scenario", "method"], columns="metric", values=["mean", "se"])
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()

order_c = ["gaussian", "clayton", "gumbel", "t", "gauss2t", "clayton2gumbel", "gauss2clayton"]
cells = (w[["copula", "scenario"]].drop_duplicates()
         .assign(o=lambda x: x.copula.map(order_c.index)).sort_values(["o", "scenario"]))
labels = [f"{c} · {s}" for c, s in zip(cells.copula, cells.scenario)]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)


fig, (a, b) = plt.subplots(1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor(SURF)
style(a)
y = np.arange(len(cells))
for k, (m, (lab, col, mk)) in enumerate(M.items()):
    s = cells.merge(w[w.method == m], on=["copula", "scenario"], how="left")
    a.errorbar(s.mean_ise_x1e4, y + (k - 2) * 0.13, xerr=2 * s.se_ise_x1e4, fmt=mk, color=col, ms=5,
               capsize=0, lw=1, label=lab)
a.set_yticks(y)
a.set_yticklabels(labels, fontsize=8, color=INK)
a.invert_yaxis()
a.set_xscale("log")
a.set_xlabel(r"copula ISE $\int(\hat C-C)^2$, whole run ($\times10^{-4}$, log)", fontsize=9, color=MUTED)
a.set_title("(a) ISE level by cell (mean ± 2 SE, 60 seeds)", fontsize=10, color=INK, loc="left")
a.legend(fontsize=7.5, frameon=False, loc="lower right")

style(b)
fb = w[w.method == "family_bank"].set_index(["copula", "scenario"])
mb = w[w.method == "mle_bank_log"].set_index(["copula", "scenario"])
for idx in fb.index:
    b.plot([1e3 * mb.loc[idx, "mean_rmse"], 1e3 * fb.loc[idx, "mean_rmse"]],
           [mb.loc[idx, "mean_ise_x1e4"], fb.loc[idx, "mean_ise_x1e4"]], color=GRID, lw=1, zorder=1)
for m in ["mle_bank_log", "family_bank"]:
    lab, col, mk = M[m]
    s = w[w.method == m]
    b.scatter(1e3 * s.mean_rmse, s.mean_ise_x1e4, color=col, marker=mk, s=36, zorder=2, label=lab,
              edgecolor=SURF, linewidth=1)
# 144*ISE >= rho error^2 holds for an estimator whose rho comes from the scored C; draw it as a guide.
xs = np.logspace(np.log10(15), np.log10(400), 50)
b.plot(xs, (xs * 1e-3) ** 2 / 144 * 1e4, color=MUTED, ls=":", lw=1,
       label=r"$\mathrm{ISE}=\mathrm{RMSE}^2/144$ (Cauchy–Schwarz floor)")
for idx in [("clayton", "constant"), ("gumbel", "constant")]:
    b.annotate(f"{idx[0]} · {idx[1]}", (1e3 * mb.loc[idx, "mean_rmse"], mb.loc[idx, "mean_ise_x1e4"]),
               fontsize=7, color=MUTED, xytext=(-6, 4), textcoords="offset points", ha="right")
b.set_xscale("log")
b.set_yscale("log")
b.set_ylim(0.03, 15)
b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
b.set_xticks([20, 50, 100, 200])
b.set_xticklabels(["20", "50", "100", "200"])
b.set_xlabel(r"$\rho_S$ RMSE, whole run ($\times10^{-3}$, log)", fontsize=9, color=MUTED)
b.set_ylabel(r"copula ISE ($\times10^{-4}$, log)", fontsize=9, color=MUTED)
b.set_title("(b) Same cell, two metrics: grey lines join the two banks", fontsize=10, color=INK, loc="left")
b.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle(r"Tracking $\rho_S$ well is not tracking the copula well", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("notes/thesis/figures/ch6_ise_levels.png", dpi=150)
print("wrote notes/thesis/figures/ch6_ise_levels.png")
