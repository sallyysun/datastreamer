"""Family bank against the online estimators, rho_S RMSE and copula CDF ISE, in the style of ch6_family_summary.png:
one point per copula x scenario cell (paired mean +- 2 SE over 60 seeds), blue = family bank better beyond 2 SE,
orange = worse, grey = within; n/19 = cells better. The offline batch estimators (oracle segment, two-sided kernel with
the best bandwidth in hindsight) see the future, so they are drawn below the line as references, untallied.
Reads results/batch_ise_42*.csv (probe_batch_ise).
Usage (from experiments/): python3 plot_batch_ise.py -> ../notes/thesis/figures/ch6_batch_ise.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fams = ["gaussian", "clayton", "gumbel", "t", "gauss2t", "clayton2gumbel", "gauss2clayton"]
d = pd.concat(pd.read_csv(f"results/batch_ise_42{'' if c == 'gaussian' else '_' + c}.csv") for c in fams)
d = d[d.window == "whole"]

rows = [("xiao_bank_log", "Xiao bank, Gaussian log loss", True), ("mle_bank_log", "MPL bank, Gaussian log loss", True),
        ("xiao_bank_own", "Xiao bank, own density", True), ("hermite_bank_own", "Hermite bank, own density", True),
        ("best_arm", "best single fixed arm (hindsight)", True),
        ("batch_segment", "reference: batch, knows the change point", False),
        ("batch_kernel", "reference: batch two-sided kernel, best h", False)]


def comparator(g, key):
    means = g.mean()
    if key == "best_arm":
        return g[means[[c for c in means.index if c.startswith("arm_")]].idxmin()]
    if key == "batch_kernel":
        return g[means[[c for c in means.index if c.startswith("batch_kernel_")]].idxmin()]
    return g[key]


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, metric, scale, title, xlab in [
        (axes[0], "rmse", 1000, r"(a) Tracking $\rho_S$", r"family bank minus competitor, RMSE ($\times10^{-3}$, symlog)"),
        (axes[1], "ise_x1e4", 1, r"(b) Whole copula: ISE $\int(\hat C-C)^2$",
         r"family bank minus competitor, ISE ($\times10^{-4}$, symlog)")]:
    w = d[d.metric == metric].pivot_table(index=["copula", "scenario", "seed"], columns="method", values="value") * scale
    cells = w.index.droplevel(-1).unique()
    for j, (key, lab, contest) in enumerate(rows):
        y0 = -j - (0.6 if not contest else 0)
        better = 0
        for i, cell in enumerate(cells):
            g = w.loc[cell]
            x = (g.family_bank - comparator(g, key)).dropna()
            m, h = x.mean(), 2 * x.std(ddof=1) / len(x) ** .5
            col = (BLUE if m + h < 0 else ORANGE if m - h > 0 else INK2) if contest else REF
            better += m + h < 0
            y = y0 + (i - (len(cells) - 1) / 2) * 0.5 / len(cells)
            ax.errorbar(m, y, xerr=h, fmt="o", ms=3, color=col, ecolor=col, elinewidth=0.8, capsize=0)
        if contest:
            ax.text(1.03, y0, f"{better}/{len(cells)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                    color=INK2)
    ax.axhline(-4.8, color=INK2, lw=0.6, ls=":")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xscale("symlog", linthresh=1 if metric != "rmse" else 10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlabel(xlab)
axes[0].set_yticks([-j - (0.6 if not c else 0) for j, (_, _, c) in enumerate(rows)], [l for _, l, _ in rows])
for t, (_, _, c) in zip(axes[0].get_yticklabels(), rows):
    if not c: t.set_color(REF)
fig.suptitle("Family bank against online competitors, 19 copula x scenario cells, 60 seeds. Left of 0 = family bank "
             "better; blue/orange = better/worse beyond 2 SE; n/19 = cells better.\nGrey rows below the line are offline "
             "references that see the future, not contestants.", fontsize=8.5, color=INK)
fig.tight_layout(rect=(0, 0, 0.97, 0.92), w_pad=3)
fig.savefig("../notes/thesis/figures/ch6_batch_ise.png", dpi=170)
print("wrote ch6_batch_ise.png")
