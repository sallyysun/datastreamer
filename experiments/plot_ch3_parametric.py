"""Chapter 3: the five parametric families of Table tab:copula-families at one Kendall tau.

fig_ch3_parametric: top row, n samples on the copula scale (U, V); bottom row, density
contours on normal scores, same levels in every panel, so equal tau with different
tails is visible directly. Self-checks: each density integrates to 1 on the grid and
each sample's tau matches the target.

Run from experiments/:
    python3 plot_ch3_parametric.py          # -> results/ch3/fig_ch3_parametric.{pdf,png}
    cp results/ch3/fig_ch3_parametric.pdf ../thesis/figures/
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, stats

from plot_ch3 import INK, MARK, note, save, scatter, tail_coefficients

TAU = 0.5
NU = 4.0
N = 1000
SEED = 42
LEVELS = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20]


def frank_tau(theta):
    debye1 = integrate.quad(lambda t: t / np.expm1(t), 0, theta)[0] / theta
    return 1 - 4 / theta * (1 - debye1)


def params(tau):
    rho = np.sin(np.pi * tau / 2)
    return {
        "gaussian": rho,
        "student_t": rho,
        "clayton": 2 * tau / (1 - tau),
        "gumbel": 1 / (1 - tau),
        "frank": optimize.brentq(lambda th: frank_tau(th) - tau, 1e-6, 100),
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
    if family == "frank":
        e = -np.expm1(-th)
        den = e - (-np.expm1(-th * u)) * (-np.expm1(-th * v))
        return np.log(th * e) - th * (u + v) - 2 * np.log(den)
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
    if family == "frank":  # conditional inversion
        u, w = rng.uniform(size=n), rng.uniform(size=n)
        v = -np.log1p(w * np.expm1(-th) / (w + (1 - w) * np.exp(-th * u))) / th
        return u, v
    raise ValueError(family)


def check(family, th, u, v):
    g = np.linspace(1e-4, 1 - 1e-4, 801)
    uu, vv = np.meshgrid(g, g)
    mass = np.trapezoid(np.trapezoid(np.exp(log_density(family, th, uu, vv)), g), g)
    tau_hat = stats.kendalltau(u, v)[0]
    se = np.sqrt(2 * (1 - TAU**2) / len(u))  # Daniels-Kendall bound on SD(tau_hat)
    assert abs(mass - 1) < 0.02, f"{family}: density integrates to {mass:.4f}"
    assert abs(tau_hat - TAU) < 4 * se, f"{family}: sample tau {tau_hat:.3f}, target {TAU}"
    return tau_hat


def fig_parametric():
    rng = np.random.default_rng(SEED)
    th = params(TAU)
    order = [("gaussian", "Gaussian", rf"$\rho$ = {th['gaussian']:.2f}"),
             ("student_t", rf"Student $t$ ($\nu$ = {NU:g})", rf"$\rho$ = {th['student_t']:.2f}"),
             ("clayton", "Clayton", rf"$\theta$ = {th['clayton']:.2f}"),
             ("gumbel", "Gumbel", rf"$\theta$ = {th['gumbel']:.2f}"),
             ("frank", "Frank", rf"$\theta$ = {th['frank']:.2f}")]

    lim = 3.2
    z = np.linspace(-lim, lim, 321)
    zx, zy = np.meshgrid(z, z)
    phi = stats.norm.pdf(zx) * stats.norm.pdf(zy)
    ramp = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", MARK, "#0d3a6e"])
    shades = [ramp(i / (len(LEVELS) - 1)) for i in range(len(LEVELS))]

    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.5))
    for j, (key, title, ptxt) in enumerate(order):
        u, v = sample(key, th[key], N, rng)
        tau_hat = check(key, th[key], u, v)
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
        note(ax, rf"$\hat\tau$ = {tau_hat:.2f}")

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
    save(fig, "fig_ch3_parametric")


if __name__ == "__main__":
    fig_parametric()
    print("wrote results/ch3/fig_ch3_parametric.{pdf,png}")
