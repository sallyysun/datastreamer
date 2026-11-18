#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// Squared Hellinger distance between a true and a forecast copula density (Chapter 6 evaluation only). Integrals over
// [0,1]^2 are taken in normal-score space, u = Phi(x), v = Phi(y), by the tensor trapezoid rule on [-L, L]^2.
namespace dmetric {

using Dens = std::function<double(double, double)>;   // a density on (0,1)^2, evaluated at (u, v)

inline double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
inline double phi(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); }
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

// Squared Hellinger distance H^2 = 1 - int sqrt(c chat): bounded in [0, 1], invariant to monotone margins.
inline double hellinger2(const Grid& g, const Table& c, const Table& ch) {
    return 1.0 - integrate(g, [&](int i, int j) { return std::sqrt(std::max(c(i, j), 0.0) * std::max(ch(i, j), 0.0)); });
}

}  // namespace dmetric
