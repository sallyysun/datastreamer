"""Chapter 3: four parametric copula families at one Spearman rho_S (Chapter 6's constant scenario): samples on the
copula scale (top) and density contours on normal scores at common levels (bottom). Parameters come from inverting each
family's rho_S by quadrature (closed form for the Gaussian); they match src/copula_family.hpp. Self-checks: each density
integrates to 1 on the grid and each sample's rho_S matches the target. Writes images/ch3_parametric.png."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from ch3_style import INK, INK_2, MARK, note, save, scatter, tail_coefficients

RHO_S = 0.6
NU = 4.0
N = 1000
SEED = 42
LEVELS = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20]


GL_U, GL_W = (lambda x, w: ((x + 1) / 2, w / 2))(*np.polynomial.legendre.leggauss(400))


def rho_s_cdf(cdf):
    """12 int int C - 3 on a Gauss-Legendre grid."""
    uu, vv = np.meshgrid(GL_U, GL_U)
    return 12 * GL_W @ cdf(uu, vv) @ GL_W - 3


def rho_s_t(r):
    """12 E[U F(Y)] - 3 with Y | X = x a scaled t_(nu+1): integrands bounded in u, so no corner mass is lost."""
    x = stats.t.ppf(GL_U, NU)[:, None]
    scale = np.sqrt((NU + x**2) * (1 - r**2) / (NU + 1))
    inner = stats.t.cdf(r * x + scale * stats.t.ppf(GL_U, NU + 1)[None, :], NU) @ GL_W
    return 12 * GL_W @ (GL_U * inner) - 3


def clayton_cdf(th):
    return lambda u, v: (u**-th + v**-th - 1) ** (-1 / th)


def gumbel_cdf(th):
    return lambda u, v: np.exp(-((-np.log(u)) ** th + (-np.log(v)) ** th) ** (1 / th))


def params(rho_s):
    solve = lambda f, lo, hi: optimize.brentq(lambda p: f(p) - rho_s, lo, hi, xtol=1e-12)
    return {
        "gaussian": 2 * np.sin(np.pi * rho_s / 6),
        "student_t": solve(rho_s_t, 0.0, 0.999),
        "clayton": solve(lambda th: rho_s_cdf(clayton_cdf(th)), 1e-3, 50),
        "gumbel": solve(lambda th: rho_s_cdf(gumbel_cdf(th)), 1 + 1e-6, 30),
    }


def log_density(family, th, u, v):
    if family == "gaussian":
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        return (-0.5 * np.log(1 - th**2)
                - (th**2 * (x**2 + y**2) - 2 * th * x * y) / (2 * (1 - th**2)))
    if family == "student_t":
        x, y = stats.t.ppf(u, NU), stats.t.ppf(v, NU)
        joint = stats.multivariate_t(shape=[[1, th], [th, 1]], df=NU)
        return joint.logpdf(np.dstack([x, y])) - stats.t.logpdf(x, NU) - stats.t.logpdf(y, NU)
    if family == "clayton":
        return (np.log1p(th) - (th + 1) * (np.log(u) + np.log(v))
                - (2 + 1 / th) * np.log(u**-th + v**-th - 1))
    if family == "gumbel":
        x, y = -np.log(u), -np.log(v)
        a = (x**th + y**th) ** (1 / th)
        return (-a - np.log(u) - np.log(v) + (th - 1) * np.log(x * y)
                + (1 - 2 * th) * np.log(a) + np.log(a + th - 1))
    raise ValueError(family)


def sample(family, th, n, rng):
    if family in ("gaussian", "student_t"):
        z = rng.multivariate_normal([0, 0], [[1, th], [th, 1]], size=n)
        if family == "gaussian":
            return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
        w = np.sqrt(rng.chisquare(NU, size=n) / NU)
        return stats.t.cdf(z[:, 0] / w, NU), stats.t.cdf(z[:, 1] / w, NU)
    e = rng.exponential(size=(n, 2))
    if family == "clayton":  # gamma frailty (Marshall-Olkin)
        s = rng.gamma(1 / th, size=(n, 1))
        uv = (1 + e / s) ** (-1 / th)
        return uv[:, 0], uv[:, 1]
    if family == "gumbel":  # positive-stable frailty, Laplace transform exp(-s^(1/th))
        a = 1 / th
        ang = rng.uniform(0, np.pi, size=(n, 1))
        w = rng.exponential(size=(n, 1))
        s = (np.sin(a * ang) / np.sin(ang) ** (1 / a)) * (np.sin((1 - a) * ang) / w) ** ((1 - a) / a)
        uv = np.exp(-((e / s) ** a))
        return uv[:, 0], uv[:, 1]
    raise ValueError(family)


def check(family, th, u, v):
    g = np.linspace(1e-4, 1 - 1e-4, 801)
    uu, vv = np.meshgrid(g, g)
    mass = np.trapezoid(np.trapezoid(np.exp(log_density(family, th, uu, vv)), g), g)
    rho_hat = stats.spearmanr(u, v)[0]
    se = (1 - RHO_S**2) * np.sqrt(1.06 / (len(u) - 3))  # Fieller-Hartley-Pearson approximation to SD(rho_hat)
    assert abs(mass - 1) < 0.02, f"{family}: density integrates to {mass:.4f}"
    assert abs(rho_hat - RHO_S) < 4 * se, f"{family}: sample rho_S {rho_hat:.3f}, target {RHO_S}"
    return rho_hat


def fig_parametric():
    rng = np.random.default_rng(SEED)
    th = params(RHO_S)
    print("parameters at rho_S = %g: " % RHO_S + ", ".join(f"{k} {v:.4f}" for k, v in th.items()))
    order = [("gaussian", "Gaussian", rf"$r$ = {th['gaussian']:.2f}"),
             ("student_t", rf"Student $t$ ($\nu$ = {NU:g})", rf"$r$ = {th['student_t']:.2f}"),
             ("clayton", "Clayton", rf"$\theta$ = {th['clayton']:.2f}"),
             ("gumbel", "Gumbel", rf"$\theta$ = {th['gumbel']:.2f}")]

    lim = 3.2
    z = np.linspace(-lim, lim, 321)
    zx, zy = np.meshgrid(z, z)
    phi = stats.norm.pdf(zx) * stats.norm.pdf(zy)
    ramp = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", MARK, "#0d3a6e"])
    shades = [ramp(i / (len(LEVELS) - 1)) for i in range(len(LEVELS))]

    fig, axes = plt.subplots(2, 4, figsize=(6.2, 3.6))
    for j, (key, title, ptxt) in enumerate(order):
        u, v = sample(key, th[key], N, rng)
        rho_hat = check(key, th[key], u, v)
        lam_l, lam_u = tail_coefficients(key, th[key], nu=NU)

        ax = axes[0, j]
        scatter(ax, u, v, size=1.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(f"{title}\n{ptxt}")
        ax.set_xlabel("$U$", labelpad=-6)
        ax.set_ylabel("$V$" if j == 0 else "", labelpad=-6)
        note(ax, rf"$\hat\rho_S$ = {rho_hat:.2f}")

        ax = axes[1, j]
        dens = np.exp(log_density(key, th[key], stats.norm.cdf(zx), stats.norm.cdf(zy))) * phi
        ax.contour(zx, zy, dens, levels=LEVELS, colors=shades, linewidths=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(r"$\Phi^{-1}(U)$")
        ax.set_ylabel(r"$\Phi^{-1}(V)$" if j == 0 else "")
        note(ax, rf"$\lambda_U$ = {lam_u:.2f}", loc="upper left")
        note(ax, rf"$\lambda_L$ = {lam_l:.2f}", loc="lower right")

    axes[0, 0].annotate("Samples", xy=(-0.62, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=8.5, color=INK)
    axes[1, 0].annotate("Density", xy=(-0.62, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=8.5, color=INK)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    fig.text(0.5, 0.01, rf"Spearman $\rho_S$ = {RHO_S} in every panel; contour levels "
             + ", ".join(f"{l:g}" for l in LEVELS) + f"; n = {N} samples.", ha="center", va="top", fontsize=7, color=INK_2)
    save(fig, "ch3_parametric")


if __name__ == "__main__":
    fig_parametric()
