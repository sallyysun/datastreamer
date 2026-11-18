#pragma once

#include <algorithm>
#include <mutex>
#include <memory>
#include <map>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "objective.hpp"
#include "special_functions.hpp"

// Parametric copula families: log-density, CDF, parameter bounds, and the Spearman conversion by Gauss-Legendre on
// the density. Clayton/Gumbel are positive-dependence only; the t copula's nu is fixed.
namespace copula {

enum class Family { Gaussian, Clayton, Gumbel, StudentT };

struct GaussLegendre {
    std::vector<double> x;
    std::vector<double> w;

    explicit GaussLegendre(int n) : x(n), w(n) {
        constexpr double kEps = 1.0e-14;
        const int m = (n + 1) / 2;
        for (int i = 0; i < m; ++i) {
            double z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
            double z1 = 0.0, pp = 0.0;
            do {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                z1 = z;
                z = z1 - p1 / pp;
            } while (std::fabs(z - z1) > kEps);
            x[i] = 0.5 * (1.0 - z);
            x[n - 1 - i] = 0.5 * (1.0 + z);
            const double weight = 1.0 / ((1.0 - z * z) * pp * pp);
            w[i] = weight;
            w[n - 1 - i] = weight;
        }
    }
};

struct Clayton {
    static constexpr double kMinParam = 1.0e-3;
    static constexpr double kMaxParam = 50.0;

    static double log_density(double u, double v, double theta) {
        const double S = std::pow(u, -theta) + std::pow(v, -theta) - 1.0;
        return std::log1p(theta)
               - (1.0 + theta) * (std::log(u) + std::log(v))
               - (2.0 + 1.0 / theta) * std::log(S);
    }

    static double cdf(double u, double v, double theta) {
        return std::pow(std::pow(u, -theta) + std::pow(v, -theta) - 1.0, -1.0 / theta);
    }

};

struct Gumbel {
    static constexpr double kMinParam = 1.0;
    static constexpr double kMaxParam = 30.0;

    static double log_density(double u, double v, double theta) {
        const double x = -std::log(u);
        const double y = -std::log(v);
        const double w = std::pow(x, theta) + std::pow(y, theta);
        const double A = std::pow(w, 1.0 / theta);
        return -A + (theta - 1.0) * (std::log(x) + std::log(y)) + (x + y)
               + (1.0 / theta - 2.0) * std::log(w) + std::log(A + theta - 1.0);
    }

    static double cdf(double u, double v, double theta) {
        const double x = -std::log(u), y = -std::log(v);
        return std::exp(-std::pow(std::pow(x, theta) + std::pow(y, theta), 1.0 / theta));
    }

};

struct Gaussian {
    static constexpr double kMinParam = -0.999;
    static constexpr double kMaxParam = 0.999;

    static double log_density(double u, double v, double rho) {
        return GaussianCopulaObjective::log_likelihood(
            special::normal_ppf(u), special::normal_ppf(v), rho);
    }
};

struct StudentT {
    static constexpr double kMinParam = -0.999;
    static constexpr double kMaxParam = 0.999;

    static double log_density(double u, double v, double rho, double nu) {
        const double x = special::t_ppf(u, nu);
        const double y = special::t_ppf(v, nu);
        const double d = 1.0 - rho * rho;
        const double Q = x * x - 2.0 * rho * x * y + y * y;
        const double log_joint =
            std::lgamma((nu + 2.0) / 2.0) - std::lgamma(nu / 2.0)
            - std::log(nu * std::numbers::pi) - 0.5 * std::log(d)
            - (nu + 2.0) / 2.0 * std::log1p(Q / (nu * d));
        const double log_mx = std::log(special::t_pdf(x, nu));
        const double log_my = std::log(special::t_pdf(y, nu));
        return log_joint - log_mx - log_my;
    }

};

namespace detail {

template <typename Density>
double spearman_integral(Density&& c) {
    static const GaussLegendre gl(48);
    double integral = 0.0;
    for (std::size_t i = 0; i < gl.x.size(); ++i) {
        const double u = gl.x[i];
        double row = 0.0;
        for (std::size_t j = 0; j < gl.x.size(); ++j) {
            const double v = gl.x[j];
            row += gl.w[j] * u * v * c(u, v);
        }
        integral += gl.w[i] * row;
    }
    return 12.0 * integral - 3.0;
}

template <typename Cdf>
double spearman_integral_cdf(Cdf&& C) {
    static const GaussLegendre gl(64);
    double integral = 0.0;
    for (std::size_t i = 0; i < gl.x.size(); ++i) {
        const double u = gl.x[i];
        double row = 0.0;
        for (std::size_t j = 0; j < gl.x.size(); ++j) row += gl.w[j] * C(u, gl.x[j]);
        integral += gl.w[i] * row;
    }
    return 12.0 * integral - 3.0;
}

struct SpearmanTable {
    std::vector<double> param;
    std::vector<double> rho_s;

    double lookup(double p) const {
        if (p <= param.front()) return rho_s.front();
        if (p >= param.back()) return rho_s.back();
        const auto it = std::upper_bound(param.begin(), param.end(), p);
        const std::size_t k = static_cast<std::size_t>(it - param.begin());
        const double t = (p - param[k - 1]) / (param[k] - param[k - 1]);
        return rho_s[k - 1] + t * (rho_s[k] - rho_s[k - 1]);
    }
};

}  // namespace detail

inline double spearman_from_param(Family family, double param, double nu = 4.0) {
    constexpr int kGrid = 96;
    switch (family) {
        case Family::Clayton: {
            static const detail::SpearmanTable table = [] {
                detail::SpearmanTable t;
                for (int k = 0; k <= kGrid; ++k) {
                    const double frac = static_cast<double>(k) / kGrid;
                    const double th = Clayton::kMinParam
                        * std::pow(Clayton::kMaxParam / Clayton::kMinParam, frac);
                    t.param.push_back(th);
                    t.rho_s.push_back(detail::spearman_integral_cdf(
                        [th](double u, double v) { return Clayton::cdf(u, v, th); }));
                }
                return t;
            }();
            return std::clamp(table.lookup(std::clamp(param, Clayton::kMinParam, Clayton::kMaxParam)), -1.0, 1.0);
        }
        case Family::Gumbel: {
            static const detail::SpearmanTable table = [] {
                detail::SpearmanTable t;
                const double gap_min = 1.0e-3, gap_max = Gumbel::kMaxParam - 1.0;
                for (int k = 0; k <= kGrid; ++k) {
                    const double frac = static_cast<double>(k) / kGrid;
                    const double th = 1.0 + gap_min * std::pow(gap_max / gap_min, frac);
                    t.param.push_back(th);
                    t.rho_s.push_back(detail::spearman_integral_cdf(
                        [th](double u, double v) { return Gumbel::cdf(u, v, th); }));
                }
                return t;
            }();
            return std::clamp(table.lookup(std::clamp(param, Gumbel::kMinParam, Gumbel::kMaxParam)), -1.0, 1.0);
        }
        case Family::StudentT: {
            // One immutable table per nu, built once under a lock and never modified afterwards; a
            // thread-local pointer skips the lock on repeat calls. The earlier version rebuilt a shared
            // static in place, a data race under threads.
            thread_local double last_nu = -1.0;
            thread_local const detail::SpearmanTable* last = nullptr;
            if (nu != last_nu || last == nullptr) {
                static std::mutex mu;
                static std::map<double, std::unique_ptr<const detail::SpearmanTable>> tables;
                std::lock_guard<std::mutex> lock(mu);
                auto& slot = tables[nu];
                if (!slot) {
                    auto t = std::make_unique<detail::SpearmanTable>();
                    for (int k = 0; k <= kGrid; ++k) {
                        const double r = StudentT::kMinParam
                            + (StudentT::kMaxParam - StudentT::kMinParam) * static_cast<double>(k) / kGrid;
                        t->param.push_back(r);
                        t->rho_s.push_back(detail::spearman_integral(
                            [r, nu](double u, double v) { return std::exp(StudentT::log_density(u, v, r, nu)); }));
                    }
                    slot = std::move(t);
                }
                last = slot.get();
                last_nu = nu;
            }
            return std::clamp(last->lookup(std::clamp(param, StudentT::kMinParam, StudentT::kMaxParam)), -1.0, 1.0);
        }
        case Family::Gaussian:
        default:
            return std::clamp(6.0 / std::numbers::pi * std::asin(param / 2.0), -1.0, 1.0);
    }
}

inline double log_density(Family family, double u, double v, double param, double nu = 4.0) {
    switch (family) {
        case Family::Clayton:  return Clayton::log_density(u, v, param);
        case Family::Gumbel:   return Gumbel::log_density(u, v, param);
        case Family::StudentT: return StudentT::log_density(u, v, param, nu);
        case Family::Gaussian:
        default:               return Gaussian::log_density(u, v, param);
    }
}

// Rank-correlation method of moments, theta = g^{-1}(rho_S) (thesis sections
// 3.5.2 and 4.3.1). Closed form for the Gaussian, bisection on the table otherwise;
// Clayton/Gumbel clamp rho_S <= 0 to their independence edge.
inline double param_from_spearman(Family family, double target, double nu = 4.0) {
    if (family == Family::Gaussian) {
        const double r = 2.0 * std::sin(std::numbers::pi * std::clamp(target, -1.0, 1.0) / 6.0);
        return std::clamp(r, Gaussian::kMinParam, Gaussian::kMaxParam);
    }
    double lo = 0.0, hi = 0.0;
    switch (family) {
        case Family::Clayton:  lo = Clayton::kMinParam;  hi = Clayton::kMaxParam;  break;
        case Family::Gumbel:   lo = Gumbel::kMinParam;   hi = Gumbel::kMaxParam;   break;
        case Family::StudentT:
        default:               lo = StudentT::kMinParam; hi = StudentT::kMaxParam; break;
    }
    const double s_lo = spearman_from_param(family, lo, nu);
    const double s_hi = spearman_from_param(family, hi, nu);
    target = std::clamp(target, std::min(s_lo, s_hi), std::max(s_lo, s_hi));
    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (spearman_from_param(family, mid, nu) < target) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

}  // namespace copula
