"""Chapter 3 background figures.

fig_ch3_margins:  one Gaussian copula under three pairs of margins -- the data
                  look different, the rank pseudo-observations are identical.
fig_ch3_families: four DGP families at the same Spearman rho -- same rank
                  dependence, different tails. Samples come from the harness's
                  own samplers via ./dump_copula_samples.

Run from this folder:
    clang++ dump_copula_samples.cpp -std=c++20 -O2 -o dump_copula_samples
    ./dump_copula_samples 0.6 2000 42
    python3 plot_ch3.py              # -> results/ch3/*.pdf, *.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = "images"
RHO_S = 0.6
SEED = 42

INK = "#0b0b0b"
INK_2 = "#52514e"
MARK = "#2a78d6"
SHADE = "#cde2fb"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.edgecolor": INK_2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": INK,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def scatter(ax, x, y, size=3.0):
    ax.scatter(x, y, s=size, color=MARK, alpha=0.45, linewidths=0, rasterized=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def note(ax, text, loc="upper left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=INK_2, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))


def pseudo(v):
    return stats.rankdata(v) / (len(v) + 1)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_margins(n=1000):
    rng = np.random.default_rng(SEED)
    rho = 2 * np.sin(np.pi * RHO_S / 6)  # Gaussian copula parameter with this rho_S
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    u, v = stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])

    margins = [
        ("Normal / Normal", stats.norm.ppf(u), stats.norm.ppf(v)),
        ("Lognormal / Exponential", stats.lognorm.ppf(u, s=1.0), stats.expon.ppf(v)),
        ("Student $t_3$ / Gamma(2)", stats.t.ppf(u, df=3), stats.gamma.ppf(v, a=2.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for j, (title, x, y) in enumerate(margins):
        ax = axes[0, j]
        scatter(ax, x, y)
        ax.set_title(title)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$" if j == 0 else "")
        r = stats.pearsonr(x, y)[0]
        rs = stats.spearmanr(x, y)[0]
        note(ax, f"Pearson $r$ = {r:.2f}\nSpearman $\\hat\\rho_S$ = {rs:.2f}")

        ax = axes[1, j]
        scatter(ax, pseudo(x), pseudo(y))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\hat U = R_X/(n+1)$")
        ax.set_ylabel(r"$\hat V = R_Y/(n+1)$" if j == 0 else "")
        note(ax, f"$\\hat\\rho_S$ = {stats.spearmanr(x, y)[0]:.2f}")

    axes[0, 0].annotate("Data", xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=8.5, color=INK)
    axes[1, 0].annotate("Pseudo-observations", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, ha="center", va="center", fontsize=8.5, color=INK)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, "fig_ch3_margins")


def tail_coefficients(family, param, nu=4.0):
    if family == "student_t":
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - param) / (1 + param)), df=nu + 1)
        return lam, lam
    if family == "clayton":
        return 2 ** (-1 / param), 0.0
    if family == "gumbel":
        return 0.0, 2 - 2 ** (1 / param)
    return 0.0, 0.0


def fig_families(path="results/ch3_copula_samples.csv"):
    df = pd.read_csv(path)
    order = [("gaussian", "Gaussian", r"$\rho$"), ("student_t", r"Student $t$ ($\nu=4$)", r"$\rho$"),
             ("clayton", "Clayton", r"$\theta$"), ("gumbel", "Gumbel", r"$\theta$")]
    q = stats.norm.ppf(0.95)
    lim = 4.0

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for j, (key, title, sym) in enumerate(order):
        d = df[df.family == key]
        z1, z2 = d.z1.to_numpy(), d.z2.to_numpy()
        param = d.param.iloc[0]
        u, v = stats.norm.cdf(z1), stats.norm.cdf(z2)

        ax = axes[0, j]
        scatter(ax, u, v, size=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{sym} = {param:.2f}")
        ax.set_xlabel("$U$")
        ax.set_ylabel("$V$" if j == 0 else "")
        rs = stats.spearmanr(u, v)[0]
        tau = stats.kendalltau(u, v)[0]
        note(ax, f"$\\hat\\rho_S$ = {rs:.2f}\n$\\hat\\tau$ = {tau:.2f}")

        ax = axes[1, j]
        ax.add_patch(plt.Rectangle((q, q), lim - q, lim - q, color=SHADE, lw=0, zorder=0))
        ax.add_patch(plt.Rectangle((-lim, -lim), lim - q, lim - q, color=SHADE, lw=0, zorder=0))
        scatter(ax, z1, z2, size=2.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\Phi^{-1}(U)$")
        ax.set_ylabel(r"$\Phi^{-1}(V)$" if j == 0 else "")
        lo = int(np.sum((z1 < -q) & (z2 < -q)))
        hi = int(np.sum((z1 > q) & (z2 > q)))
        lam_l, lam_u = tail_coefficients(key, param)
        note(ax, f"$\\lambda_U$ = {lam_u:.2f}\nboth > 95%: {hi}", loc="upper left")
        note(ax, f"$\\lambda_L$ = {lam_l:.2f}\nboth < 5%: {lo}", loc="lower right")

    axes[0, 0].annotate("Copula scale", xy=(-0.5, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=8.5)
    axes[1, 0].annotate("Normal scores", xy=(-0.5, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=8.5)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, "fig_ch3_families")


if __name__ == "__main__":
    fig_margins()
    fig_families()
    print(f"wrote {OUT}/fig_ch3_margins.{{pdf,png}} and {OUT}/fig_ch3_families.{{pdf,png}}")
