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

from plot_ch3 import INK, INK_2, MARK, SHADE, save

N = 12
RHO = 0.7
SEED = 20261025
V_SLICE = 0.55
C_EMP, C_CHK, C_BETA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e4e3df"


def sample():
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((N, 2))
    x = z[:, 0]
    y = RHO * z[:, 0] + np.sqrt(1 - RHO**2) * z[:, 1]
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


def unit_square(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("$u$")
    ax.set_ylabel("$v$")
    ax.set_title(title, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rank_grid(ax, zorder=0):
    for k in range(1, N):
        ax.axhline(k / N, color=GRID, lw=0.5, zorder=zorder)
        ax.axvline(k / N, color=GRID, lw=0.5, zorder=zorder)


def fig_empirical():
    r, s = sample()
    u_line = np.linspace(0, 1, 1201)  # includes every k/12
    checks(r, s, u_line)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.82], hspace=0.45, wspace=1.4)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, 2 * j:2 * j + 2]) for j in range(3))
    ax_d, ax_e = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])

    unit_square(ax_a, "A  Empirical copula")
    rank_grid(ax_a)
    for ri, si in zip(r, s):
        ax_a.add_patch(Rectangle(((ri - 1) / N, (si - 1) / N), 1 / N, 1 / N, fc=SHADE, ec="none", zorder=1))
    ax_a.scatter(r / N, s / N, s=16, color=C_EMP, zorder=3, linewidths=0, clip_on=False)
    ax_a.text(0.97, 0.03, "mass 1/12 per point", transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=6.5, color=INK_2)

    unit_square(ax_b, "B  Checkerboard density")
    cells = np.zeros((N, N))
    cells[s - 1, r - 1] = N
    ax_b.imshow(cells, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap(["white", C_CHK]),
                vmin=0, vmax=N, interpolation="nearest", zorder=1)
    rank_grid(ax_b, zorder=2)
    ax_b.legend(handles=[Patch(fc="white", ec=INK_2, lw=0.5, label="0"),
                         Patch(fc=C_CHK, label="12")], title="density", loc="upper left",
                fontsize=6.5, title_fontsize=6.5, frameon=False, handlelength=1.0,
                borderaxespad=0.1)

    unit_square(ax_c, "C  Beta density")
    t = np.linspace(0, 1, 241)
    tu, tv = np.meshgrid(t, t)
    dens = np.mean([beta_pdf(ri, tu) * beta_pdf(si, tv) for ri, si in zip(r, s)], axis=0)
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax_c.imshow(dens, origin="lower", extent=(0, 1, 0, 1), cmap=seq, vmin=0, vmax=3,
                     interpolation="bilinear")
    cb = fig.colorbar(im, cax=ax_c.inset_axes([1.06, 0.0, 0.05, 1.0]),
                      extend="max" if dens.max() > 3 else "neither")
    cb.set_ticks([0, 1, 2, 3])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5)

    def cdf_panel(ax, v, title, ylabel):
        u_step = np.r_[0, np.sort(r) / N, 1]
        ax.step(u_step, c_emp(r, s, u_step, v), where="post", color=C_EMP, lw=1.6, label="empirical")
        ax.plot(u_line, c_chk(r, s, u_line, v), color=C_CHK, lw=1.6, label="checkerboard")
        ax.plot(u_line, c_beta(r, s, u_line, v), color=C_BETA, lw=1.6, ls=(0, (4, 2)), label="beta")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0 if v == 1 else None)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel("$u$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    cdf_panel(ax_d, V_SLICE, f"D  CDFs at $v$ = {V_SLICE:g}", f"$C(u, {V_SLICE:g})$")
    cdf_panel(ax_e, 1.0, "E  First margin, $v$ = 1", "$C(u, 1)$")
    ax_d.legend(loc="upper left", frameon=False, fontsize=7)

    save(fig, "fig_ch3_empirical")


if __name__ == "__main__":
    fig_empirical()
    print("wrote results/ch3/fig_ch3_empirical.{pdf,png}")
