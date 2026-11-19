"""Chapter 3: the empirical and empirical checkerboard copulas of one small sample (n = 12, latent Gaussian correlation
0.7): A rank pairs with mass 1/n, B checkerboard density, C both copulas at v = 0.55, D both first margins. Self-checks:
the checkerboard density integrates to 1, has uniform margins and equals the empirical copula at every grid vertex.
Writes images/ch3_empirical.png."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from ch3_style import GRID, OUT, SHADE, note, plt, save

N, RHO, SEED, V_SLICE = 12, 0.7, 20261025, 0.55
C_EMP, C_CHK = "#2a78d6", "#eb6834"


def sample():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N)
    y = RHO * x + np.sqrt(1 - RHO**2) * rng.standard_normal(N)
    r, s = x.argsort().argsort() + 1, y.argsort().argsort() + 1   # ranks 1..n (continuous data, no ties)
    assert len(set(r)) == N and len(set(s)) == N
    return r, s


def b_cell(rank, t):
    return np.clip(N * t - rank + 1, 0.0, 1.0)


def c_emp(r, s, u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.mean([(ri / N <= u) & (si / N <= v) for ri, si in zip(r, s)], axis=0)


def c_chk(r, s, u, v):
    return np.mean([b_cell(ri, u) * b_cell(si, v) for ri, si in zip(r, s)], axis=0)


def checks(r, s, u_line):
    cells = np.zeros((N, N))
    cells[s - 1, r - 1] = N
    assert abs(cells.sum() / N**2 - 1) < 1e-15
    assert max(np.abs(c_chk(r, s, u_line, 1.0) - u_line).max(), np.abs(c_chk(r, s, 1.0, u_line) - u_line).max()) < 1e-12
    g = np.arange(N + 1) / N
    gu, gv = np.meshgrid(g, g)
    assert np.abs(c_chk(r, s, gu, gv) - c_emp(r, s, gu, gv)).max() < 1e-15


def rank_grid(ax, zorder=0):
    for k in range(1, N):
        ax.axhline(k / N, color=GRID, lw=0.4, zorder=zorder)
        ax.axvline(k / N, color=GRID, lw=0.4, zorder=zorder)


def unit_square(ax, title, ylabel=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("$u$", labelpad=-6)
    ax.set_ylabel("$v$" if ylabel else "", labelpad=-6)
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def line_panel(ax, title, ylabel):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("$u$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    r, s = sample()
    u_line = np.linspace(0, 1, 1201)
    checks(r, s, u_line)
    fig = plt.figure(figsize=(6.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], hspace=0.45, wspace=0.35)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))

    unit_square(ax_a, "A  Empirical copula\npoint mass $1/n$ each")
    rank_grid(ax_a)
    for ri, si in zip(r, s):
        ax_a.add_patch(Rectangle(((ri - 1) / N, (si - 1) / N), 1 / N, 1 / N, fc=SHADE, ec="none", zorder=1))
    ax_a.scatter(r / N, s / N, s=12, color=C_EMP, zorder=3, linewidths=0, clip_on=False)
    note(ax_a, rf"$\hat\rho_S$ = {1 - 6 * np.sum((r - s) ** 2) / (N * (N**2 - 1)):.2f}")

    unit_square(ax_b, "B  Checkerboard\ndensity $n$ on occupied cells", ylabel=False)
    cells = np.zeros((N, N))
    cells[s - 1, r - 1] = N
    ax_b.imshow(cells, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap(["white", C_CHK]), vmin=0, vmax=N,
                interpolation="nearest", alpha=0.85, zorder=1)
    rank_grid(ax_b, zorder=2)

    u_step = np.r_[0, np.sort(r) / N, 1]
    for ax, v in ((ax_c, V_SLICE), (ax_d, 1.0)):
        ax.step(u_step, c_emp(r, s, u_step, v), where="post", color=C_EMP, lw=1.2, label="empirical")
        ax.plot(u_line, c_chk(r, s, u_line, v), color=C_CHK, lw=1.2, label="checkerboard")
    line_panel(ax_c, f"C  Copulas at $v$ = {V_SLICE:g}", f"$C(u, {V_SLICE:g})$")
    ax_c.set_ylim(0, 0.6)
    ax_c.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.2)
    note(ax_c, "checkerboard = empirical\nat every grid vertex")
    line_panel(ax_d, "D  First margin, $v$ = 1", "$C(u, 1)$")
    ax_d.set_ylim(0, 1)
    note(ax_d, "checkerboard: exactly $u$\nempirical: $\\lfloor nu \\rfloor / n$")

    save(fig, "ch3_empirical")


if __name__ == "__main__":
    main()
