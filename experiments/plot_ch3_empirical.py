"""Chapter 3: empirical, empirical checkerboard and empirical beta copulas on one small sample.

fig_ch3_empirical, n = 12 pairs, latent Gaussian correlation 0.7, seed 20261025:
  A  scaled rank pairs (R_i/n, S_i/n), mass 1/n each, at the upper-right corner of their cell
  B  checkerboard density: n on each occupied cell, 0 elsewhere
  C  beta density (1/n) sum_i f_{R_i}(u) f_{S_i}(v), f_r the Beta(r, n+1-r) density
  D  the three CDFs at v = 0.55
  E  the three first margins (v = 1): floor(nu)/n for the empirical copula, u for the other two
Self-checks: both densities integrate to 1, checkerboard and beta margins are uniform,
the checkerboard equals the empirical copula at every grid vertex.

Run from experiments/:
    python3 plot_ch3_empirical.py           # -> results/ch3/fig_ch3_empirical.{pdf,png}
    cp results/ch3/fig_ch3_empirical.pdf ../thesis/figures/
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy import special, stats

from plot_ch3 import INK_2, SHADE, save

N = 12
RHO = 0.7
SEED = 20261025
V_SLICE = 0.55
C_EMP, C_CHK, C_BETA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e4e3df"


def sample():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N)
    y = RHO * x + np.sqrt(1 - RHO**2) * rng.standard_normal(N)
    r = stats.rankdata(x).astype(int)
    s = stats.rankdata(y).astype(int)
    assert len(set(r)) == N and len(set(s)) == N
    return r, s


def b_cell(rank, t):
    return np.clip(N * t - rank + 1, 0.0, 1.0)


def beta_cdf(rank, t):
    k = np.arange(rank, N + 1)
    t = np.asarray(t, dtype=float)[..., None]
    return (special.comb(N, k) * t**k * (1 - t) ** (N - k)).sum(axis=-1)


def beta_pdf(rank, t):
    return stats.beta.pdf(t, rank, N + 1 - rank)


def c_emp(r, s, u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.mean([(ri / N <= u) & (si / N <= v) for ri, si in zip(r, s)], axis=0)


def c_chk(r, s, u, v):
    return np.mean([b_cell(ri, u) * b_cell(si, v) for ri, si in zip(r, s)], axis=0)


def c_beta(r, s, u, v):
    return np.mean([beta_cdf(ri, u) * beta_cdf(si, v) for ri, si in zip(r, s)], axis=0)


def checks(r, s, u_line):
    x, w = np.polynomial.legendre.leggauss(64)
    x, w = (x + 1) / 2, w / 2
    beta_mass = sum(w @ beta_pdf(ri, x) * (w @ beta_pdf(si, x)) for ri, si in zip(r, s)) / N
    cells = np.zeros((N, N))
    cells[s - 1, r - 1] = N
    chk_mass = cells.sum() / N**2
    assert abs(beta_mass - 1) < 1e-12 and abs(chk_mass - 1) < 1e-15

    margin_err = max(np.abs(c_chk(r, s, u_line, 1.0) - u_line).max(),
                     np.abs(c_chk(r, s, 1.0, u_line) - u_line).max(),
                     np.abs(c_beta(r, s, u_line, 1.0) - u_line).max(),
                     np.abs(c_beta(r, s, 1.0, u_line) - u_line).max())
    assert margin_err < 1e-12

    g = np.arange(N + 1) / N
    gu, gv = np.meshgrid(g, g)
    vertex_gap = np.abs(c_chk(r, s, gu, gv) - c_emp(r, s, gu, gv)).max()
    assert vertex_gap < 1e-15
    beta_gap = np.abs(c_beta(r, s, gu, gv) - c_emp(r, s, gu, gv)).max()
    print(f"beta mass {beta_mass:.16f}; max margin error {margin_err:.1e}; "
          f"checkerboard vs empirical at {gu.size} vertices {vertex_gap:.1e}; "
          f"beta vs empirical at vertices up to {beta_gap:.3f}")


TICKS = [0, 0.25, 0.5, 0.75, 1]
TICKLABELS = ["0", ".25", ".5", ".75", "1"]


def rank_grid(ax):
    for k in range(1, N):
        ax.axhline(k / N, color=GRID, lw=0.5, zorder=0)
        ax.axvline(k / N, color=GRID, lw=0.5, zorder=0)


def unit_square(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks(TICKS, TICKLABELS)
    ax.set_yticks(TICKS, TICKLABELS)
    ax.set_xlabel("$u$")
    ax.set_ylabel("$v$")
    ax.set_title(title, loc="left", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_empirical():
    r, s = sample()
    u_line = np.linspace(0, 1, 1201)  # includes every k/12
    checks(r, s, u_line)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.82], hspace=0.45, wspace=1.4)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, 2 * j:2 * j + 2]) for j in range(3))
    ax_d, ax_e = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])

    unit_square(ax_a, "A  Empirical point masses")
    rank_grid(ax_a)
    for ri, si in zip(r, s):
        ax_a.add_patch(Rectangle(((ri - 1) / N, (si - 1) / N), 1 / N, 1 / N, fc=SHADE, ec="none", zorder=1))
    ax_a.scatter(r / N, s / N, s=16, color=C_EMP, zorder=3, linewidths=0, clip_on=False,
                 label="Point mass $1/n$ each")

    unit_square(ax_b, "B  Checkerboard density")
    cells = np.zeros((N, N))
    cells[s - 1, r - 1] = N
    ax_b.imshow(cells, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap(["white", C_CHK]),
                vmin=0, vmax=N, interpolation="nearest", zorder=1)

    unit_square(ax_c, "C  Empirical beta density")
    t = np.linspace(0, 1, 241)
    tu, tv = np.meshgrid(t, t)
    dens = np.mean([beta_pdf(ri, tu) * beta_pdf(si, tv) for ri, si in zip(r, s)], axis=0)
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ecdc", "#56c596", C_BETA, "#0b5e3f"])
    im = ax_c.imshow(dens, origin="lower", extent=(0, 1, 0, 1), cmap=seq, vmin=0, vmax=3,
                     interpolation="bilinear")
    cb = fig.colorbar(im, cax=ax_c.inset_axes([1.06, 0.0, 0.05, 1.0]),
                      extend="max" if dens.max() > 3 else "neither")
    cb.set_ticks([0, 1, 2, 3])
    cb.set_label("Beta density", fontsize=6.5, color=INK_2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5)

    for ax, handles in ((ax_a, [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=C_EMP,
                                           label="Point mass $1/n$ each")]),
                        (ax_b, [Patch(fc=C_CHK, label=r"$c_n^{\#} = n = 12$"),
                                Patch(fc="white", ec=INK_2, lw=0.5, label=r"$c_n^{\#} = 0$")]),
                        (ax_c, [Patch(fc=C_BETA, label=r"$c_n^{\beta}(u,v)$: smooth, nonnegative")])):
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
                  fontsize=6.5, handlelength=1.0, borderaxespad=0.1, columnspacing=1.0)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=16)

    def cdf_panel(ax, v, title, labels):
        u_step = np.r_[0, np.sort(r) / N, 1]
        ax.step(u_step, c_emp(r, s, u_step, v), where="post", color=C_EMP, lw=1.6, label=labels[0])
        ax.plot(u_line, c_chk(r, s, u_line, v), color=C_CHK, lw=1.6, label=labels[1])
        ax.plot(u_line, c_beta(r, s, u_line, v), color=C_BETA, lw=1.6, ls=(0, (4, 2)), label=labels[2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0 if v == 1 else None)
        ax.set_xticks(TICKS, TICKLABELS)
        ax.set_xlabel("$u$")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.legend(loc="lower right" if v < 1 else "upper left", frameon=False, fontsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    cdf_panel(ax_d, V_SLICE, f"D  Joint CDF slice at $v$ = {V_SLICE:g}",
              [rf"Empirical $C_n(u, {V_SLICE:g})$", rf"Checkerboard $C_n^{{\#}}(u, {V_SLICE:g})$",
               rf"Empirical beta $C_n^{{\beta}}(u, {V_SLICE:g})$"])
    cdf_panel(ax_e, 1.0, "E  Marginal CDF at $v$ = 1",
              [r"Empirical: $\lfloor nu \rfloor / n$", "Checkerboard: $u$", "Empirical beta: $u$"])

    save(fig, "fig_ch3_empirical")


if __name__ == "__main__":
    fig_empirical()
    print("wrote results/ch3/fig_ch3_empirical.{pdf,png}")
