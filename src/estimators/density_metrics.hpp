#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// Evaluation-only density metrics against a known true copula density (Chapter 6). Nothing here is used by
// an estimator. All integrals over [0,1]^2 are taken in normal-score space, u = Phi(x), v = Phi(y):
//   int g(u,v) du dv = int g(Phi x, Phi y) phi(x) phi(y) dx dy,
// by the tensor trapezoid rule on [-L, L]^2 (or on a truncated square for interior quantities).
// Definitions, feasibility and the protocol: archive/notes/studies/chapter6_metrics.md.
namespace dmetric {

using Dens = std::function<double(double, double)>;   // a density on (0,1)^2, evaluated at (u, v)

inline double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
inline double phi(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); }
inline double probit(double p) {   // bisection: exact to double precision, only used to build grids
    double lo = -40.0, hi = 40.0;
    for (int i = 0; i < 200; ++i) { const double m = 0.5 * (lo + hi); (Phi(m) < p ? lo : hi) = m; }
    return 0.5 * (lo + hi);
}

// Nodes z_i on [a, b], u_i = Phi(z_i), weights w_i = h * trapezoid * phi(z_i) (so sum_i w_i g(u_i) ~ int g du).
struct Grid {
    std::vector<double> z, u, w;
    Grid(int n, double a, double b) : z(n), u(n), w(n) {
        const double h = (b - a) / (n - 1);
        for (int i = 0; i < n; ++i) {
            z[i] = a + h * i;
            u[i] = Phi(z[i]);
            w[i] = h * (i == 0 || i == n - 1 ? 0.5 : 1.0) * phi(z[i]);
        }
    }
    static Grid full(int n, double L) { return Grid(n, -L, L); }
    static Grid interior(int n, double eps) { const double z = probit(1.0 - eps); return Grid(n, -z, z); }
    int size() const { return int(z.size()); }
};

// Tabulates a density once on a grid, so several metrics reuse the same evaluations.
struct Table {
    std::vector<double> v;
    int n;
    Table(const Grid& g, const Dens& c) : v(std::size_t(g.size()) * g.size()), n(g.size()) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) v[std::size_t(i) * n + j] = c(g.u[i], g.u[j]);
    }
    double operator()(int i, int j) const { return v[std::size_t(i) * n + j]; }
};

template <class F> double integrate(const Grid& g, F f) {   // f(i, j) -> integrand at node (i, j), du dv measure
    double s = 0.0;
    for (int i = 0; i < g.size(); ++i)
        for (int j = 0; j < g.size(); ++j) s += f(i, j) * g.w[i] * g.w[j];
    return s;
}

inline double mass(const Grid& g, const Table& a) { return integrate(g, [&](int i, int j) { return a(i, j); }); }

// Squared Hellinger distance H^2 = 1 - int sqrt(c chat): bounded in [0, 1], invariant to monotone margins.
inline double hellinger2(const Grid& g, const Table& c, const Table& ch) {
    return 1.0 - integrate(g, [&](int i, int j) { return std::sqrt(std::max(c(i, j), 0.0) * std::max(ch(i, j), 0.0)); });
}

// Copula-scale ISE int (c - chat)^2 du dv over the grid's square. On a full grid it approximates the
// full-domain ISE only where that integral is finite (chapter6_metrics.md §3); on an interior grid it is
// the interior ISE on [eps, 1 - eps]^2.
inline double ise(const Grid& g, const Table& c, const Table& ch) {
    return integrate(g, [&](int i, int j) { const double e = c(i, j) - ch(i, j); return e * e; });
}

// Normal-score ISE int (f - fhat)^2 dx dy = int (c - chat)^2 phi(z_u) phi(z_v) du dv (finite for every tested pair).
inline double ise_z(const Grid& g, const Table& c, const Table& ch) {
    return integrate(g, [&](int i, int j) {
        const double e = c(i, j) - ch(i, j);
        return e * e * phi(g.z[i]) * phi(g.z[j]);
    });
}

// KL(c || chat) = int c log(c / chat); the expectation of the excess log loss at the true PIT.
inline double kl(const Grid& g, const Table& c, const Table& ch) {
    return integrate(g, [&](int i, int j) {
        const double a = c(i, j), b = ch(i, j);
        return a > 0.0 ? a * std::log(a / std::max(b, 1e-300)) : 0.0;
    });
}

// Marginal check of a bivariate density on the unit square: m(u) = int chat(u, v) dv should be 1 for a
// copula. Returns max_u |m(u) - 1| over the grid's nodes and int |m(u) - 1| du (both margins: the worse).
struct MarginCheck { double max_dev, l1_dev; };
inline MarginCheck margins(const Grid& g, const Table& ch) {
    MarginCheck r{0.0, 0.0};
    for (int axis = 0; axis < 2; ++axis) {
        double mx = 0.0, l1 = 0.0;
        for (int i = 0; i < g.size(); ++i) {
            double m = 0.0;
            for (int j = 0; j < g.size(); ++j) m += (axis == 0 ? ch(i, j) : ch(j, i)) * g.w[j];
            mx = std::max(mx, std::abs(m - 1.0));
            l1 += std::abs(m - 1.0) * g.w[i];
        }
        r.max_dev = std::max(r.max_dev, mx);
        r.l1_dev = std::max(r.l1_dev, l1);
    }
    return r;
}

// Closed forms for two Gaussian copulas (parameters r, s): checks of the quadrature. int c_r c_s = 1/(1 - r s)
// by the Mehler expansion, so ISE = 1/(1-r^2) - 2/(1-rs) + 1/(1-s^2).
inline double gauss_h2(double r, double s) {
    return 1.0 - std::pow((1 - r * r) * (1 - s * s), 0.25) / std::sqrt(1 - 0.25 * (r + s) * (r + s));
}
inline double gauss_kl(double r, double s) {
    return 0.5 * (std::log((1 - s * s) / (1 - r * r)) + (2 - 2 * r * s) / (1 - s * s) - 2);
}
inline double gauss_ise(double r, double s) { return 1 / (1 - r * r) - 2 / (1 - r * s) + 1 / (1 - s * s); }

}  // namespace dmetric
