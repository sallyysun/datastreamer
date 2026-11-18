#pragma once

// Chapter 6 strategies, scoring windows and density metrics shared by experiments and the consumer.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "study_config.hpp"
#include "../copula_family.hpp"
#include "dependence/mff.hpp"
#include "dependence/vff.hpp"
#include "registry.hpp"

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

namespace thesis {

inline const std::vector<double>& vff_rates() {
    static const std::vector<double> rates{kVffSlow, kVffFast};
    return rates;
}
// Strategy order of every output row: five fixed arms, MFF, VFF at each rate.
inline const std::vector<std::string>& strategy_names() {
    static const auto names = [] {
        std::vector<std::string> result;
        for (double memory : kMem) result.push_back("fixed_" + std::to_string(static_cast<int>(memory)));
        result.insert(result.end(), {"mff", "vff_slow", "vff_fast"});
        return result;
    }();
    return names;
}

using copula::family_of;

inline bool scorable(const std::string& family) {
    return copula::family_from_name(family).has_value();
}

inline double true_param(const std::string& family, double target) {
    return copula::param_from_spearman(family_of(family), target);
}

struct Win { const char* name; std::size_t a, b; };
inline std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") {
        w.push_back({"pre", kBurn, kChange}); w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500}); w.push_back({"post", kChange, kN});
    }
    return w;
}
// H^2 checkpoints: every 50 from kBurn, and every 10 in the 100 observations after the jump.
inline bool checkpoint(const std::string& sc, std::size_t t) {
    return t >= kBurn && ((t - kBurn) % 50 == 0 || (sc == "jump" && t >= kChange && t < kChange + 100 && (t - kChange) % 10 == 0));
}

inline double gauss_logc(double rho, double u, double v) {
    rho = std::clamp(rho, -0.999, 0.999);
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

// H^2(c_true(par), c_G(rho)) tabulated over rho in [-0.99, 0.99] (step 0.0025) once per truth parameter.
struct H2Table {
    const dmetric::Grid g = dmetric::Grid::full(81, 8.0);
    copula::Family F;
    std::map<double, std::vector<double>> cache;
    static constexpr double lo = -0.99, step = 0.0025;
    static constexpr int n = 793;
    double direct(double par, double rho) const {
        const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(rho, a, b)); });
        return dmetric::hellinger2(g, truth, f);
    }
    // H^2(c_true(par), sum_k w_k c_G(rho_k)) on the same grid, computed directly (no rho interpolation).
    std::map<double, std::vector<double>> truth;
    double mix(double par, const double* rho, const double* w, std::size_t K) {
        auto it = truth.find(par);
        if (it == truth.end()) {
            const dmetric::Table t(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            it = truth.emplace(par, t.v).first;
        }
        const int n = g.size();
        std::vector<double> r(K), c0(K), c1(K);
        for (std::size_t k = 0; k < K; ++k) {
            r[k] = copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rho[k], -0.999, 0.999));
            c0[k] = 1 / std::sqrt(1 - r[k] * r[k]); c1[k] = 1 / (2 * (1 - r[k] * r[k]));
        }
        double s = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double x = g.z[i], y = g.z[j];
                double m = 0;
                for (std::size_t k = 0; k < K; ++k)
                    if (w[k] > 0) m += w[k] * c0[k] * std::exp(-(r[k] * r[k] * (x * x + y * y) - 2 * r[k] * x * y) * c1[k]);
                s += std::sqrt(std::max(it->second[std::size_t(i) * n + j], 0.0) * m) * g.w[i] * g.w[j];
            }
        return 1.0 - s;
    }
    // log sum_k w_k c_G(rho_k)(u, v), each component as gauss_logc
    static double mix_logc(const double* rho, const double* w, std::size_t K, double u, double v) {
        double top = -INFINITY;
        std::vector<double> l(K);
        for (std::size_t k = 0; k < K; ++k) { l[k] = w[k] > 0 ? std::log(w[k]) + gauss_logc(rho[k], u, v) : -INFINITY; top = std::max(top, l[k]); }
        double s = 0;
        for (double x : l) s += std::exp(x - top);
        return top + std::log(s);
    }
    double operator()(double par, double rho) {
        auto it = cache.find(par);
        if (it == cache.end()) {
            const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            std::vector<double> tab(n);
            for (int i = 0; i < n; ++i) {
                const double r = lo + step * i;
                const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(r, a, b)); });
                tab[i] = dmetric::hellinger2(g, truth, f);
            }
            it = cache.emplace(par, std::move(tab)).first;
        }
        const double x = (std::clamp(rho, lo, lo + step * (n - 1)) - lo) / step;
        const int i = std::min(n - 2, int(x));
        return it->second[i] + (x - i) * (it->second[i + 1] - it->second[i]);
    }
};

struct ScoringObservation {
    double rho, u, v, parameter, log_density;
    bool hellinger_checkpoint;
};

struct ForecastScore { double squared_error, excess, h2; };

inline ForecastScore score_forecast(H2Table& h2, const ScoringObservation& truth, std::span<const double> forecasts,
                                    std::size_t strategy, std::span<const double> mixture_weights = {}) {
    const double rho = forecasts[strategy];
    const double error = rho - truth.rho;
    if (strategy == kK && !mixture_weights.empty()) {
        const auto* weights = mixture_weights.data();
        return {error * error,
                truth.log_density - H2Table::mix_logc(forecasts.data(), weights, kK, truth.u, truth.v),
                truth.hellinger_checkpoint ? h2.mix(truth.parameter, forecasts.data(), weights, kK) : NAN};
    }
    return {error * error, truth.log_density - gauss_logc(rho, truth.u, truth.v),
            truth.hellinger_checkpoint ? h2(truth.parameter, rho) : NAN};
}

}  // namespace thesis
