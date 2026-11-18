#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "special_functions.hpp"

// Parametric copula families: log-density, CDF, parameter bounds, and the conversion between each family's parameter
// and Spearman's rho (closed form for the Gaussian, tabulated otherwise). Clayton/Gumbel are positive-dependence only;
// the t copula has nu = 4.
namespace copula {

enum class Family { Gaussian, Clayton, Gumbel, StudentT };

inline std::optional<Family> family_from_name(std::string_view name) {
    if (name == "gaussian") return Family::Gaussian;
    if (name == "clayton") return Family::Clayton;
    if (name == "gumbel") return Family::Gumbel;
    if (name == "t" || name == "student_t") return Family::StudentT;
    return std::nullopt;
}

inline Family family_of(std::string_view name) {
    return family_from_name(name).value_or(Family::Gaussian);
}

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
        const double zx = special::normal_ppf(u), zy = special::normal_ppf(v);
        const double d = 1.0 - rho * rho;
        const double q = rho * rho * (zx * zx + zy * zy) - 2.0 * rho * zx * zy;
        return -0.5 * std::log(d) - q / (2.0 * d);
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
    double min_param, max_param;
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

    double spearman(double parameter) const {
        return std::clamp(lookup(std::clamp(parameter, min_param, max_param)), -1.0, 1.0);
    }

    // The table is increasing (checked when it is built), so the inverse interpolates it the other way round.
    double inverse(double target) const {
        const double lo = min_param, hi = max_param;
        const double rho_lo = spearman(lo), rho_hi = spearman(hi);
        target = std::clamp(target, rho_lo, rho_hi);
        // Gumbel's independence boundary lies below its first tabulated point.
        if (target <= rho_lo || std::isnan(target)) return lo;
        const auto upper = std::lower_bound(rho_s.begin(), rho_s.end(), target);
        if (upper == rho_s.end()) return hi;
        const std::size_t k = static_cast<std::size_t>(upper - rho_s.begin());
        if (k == 0) return lo;
        const double t = (target - rho_s[k - 1]) / (rho_s[k] - rho_s[k - 1]);
        return std::clamp(param[k - 1] + t * (param[k] - param[k - 1]), lo, hi);
    }

    const SpearmanTable& checked() const {
        if (!std::all_of(rho_s.begin(), rho_s.end(), [](double rho) { return std::isfinite(rho); })
            || !std::is_sorted(rho_s.begin(), rho_s.end()))
            throw std::logic_error("Spearman table is not finite and increasing");
        return *this;
    }
};

template <typename Parameter, typename RankCorrelation>
SpearmanTable build_spearman_table(double lo, double hi, Parameter parameter, RankCorrelation rank_correlation) {
    constexpr int kGrid = 96;
    SpearmanTable table{lo, hi, {}, {}};
    table.param.reserve(kGrid + 1);
    table.rho_s.reserve(kGrid + 1);
    for (int k = 0; k <= kGrid; ++k) {
        const double p = parameter(k, kGrid);
        table.param.push_back(p);
        table.rho_s.push_back(rank_correlation(p));
    }
    table.checked();
    return table;
}

// Spearman's rho of the t copula. A density integral on the unit square misses its corner mass (0.0024 too high at
// r = 0.81, nu = 4), so this uses the exact representation rho_S = (6/pi) E[asin(r / sqrt((1 + W0/W1)(1 + W0/W2)))],
// (W0, W1, W2) ~ Dirichlet(nu/2, nu/2, nu/2), written as W0 = A, W1 = (1 - A) B, W2 = (1 - A)(1 - B) with
// A ~ Beta(nu/2, nu), B ~ Beta(nu/2, nu/2), by Gauss-Legendre on [0, 1]^2; odd in r, tabulated on 4001 points.
inline SpearmanTable student_t_spearman_table(double nu) {
    static const GaussLegendre gl(256);
    const std::size_t n = gl.x.size();
    const double a = nu / 2.0;
    const double ca = std::lgamma(3.0 * a) - std::lgamma(a) - std::lgamma(2.0 * a);
    const double cb = std::lgamma(2.0 * a) - 2.0 * std::lgamma(a);
    std::vector<double> wa(n), wb(n), scale(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = gl.x[i];
        wa[i] = gl.w[i] * std::exp(ca + (a - 1.0) * std::log(x) + (2.0 * a - 1.0) * std::log1p(-x));
        wb[i] = gl.w[i] * std::exp(cb + (a - 1.0) * (std::log(x) + std::log1p(-x)));
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double A = gl.x[i], B = gl.x[j];
            const double x = A / ((1.0 - A) * B), y = A / ((1.0 - A) * (1.0 - B));
            scale[i * n + j] = 1.0 / std::sqrt((1.0 + x) * (1.0 + y));
        }
    constexpr int kHalf = 2000;
    SpearmanTable t{StudentT::kMinParam, StudentT::kMaxParam, std::vector<double>(2 * kHalf + 1), std::vector<double>(2 * kHalf + 1)};
    for (int k = 0; k <= kHalf; ++k) {
        const double r = StudentT::kMaxParam * k / kHalf;
        double e = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double row = 0.0;
            for (std::size_t j = 0; j < n; ++j) row += wb[j] * std::asin(r * scale[i * n + j]);
            e += wa[i] * row;
        }
        const double rho = 6.0 / std::numbers::pi * e;
        t.param[kHalf + k] = r;
        t.rho_s[kHalf + k] = rho;
        t.param[kHalf - k] = -r;
        t.rho_s[kHalf - k] = -rho;
    }
    t.checked();
    return t;
}

inline const SpearmanTable& spearman_table(Family family, double nu) {
    if (family == Family::Clayton) {
        static const auto table = build_spearman_table(
            Clayton::kMinParam, Clayton::kMaxParam,
            [](int k, int grid) {
                const double fraction = static_cast<double>(k) / grid;
                return Clayton::kMinParam * std::pow(Clayton::kMaxParam / Clayton::kMinParam, fraction);
            },
            [](double theta) {
                return spearman_integral_cdf([theta](double u, double v) { return Clayton::cdf(u, v, theta); });
            });
        return table;
    }
    if (family == Family::Gumbel) {
        static const auto table = build_spearman_table(
            Gumbel::kMinParam, Gumbel::kMaxParam,
            [](int k, int grid) {
                const double gap_min = 1.0e-3, gap_max = Gumbel::kMaxParam - 1.0;
                const double fraction = static_cast<double>(k) / grid;
                return 1.0 + gap_min * std::pow(gap_max / gap_min, fraction);
            },
            [](double theta) {
                return spearman_integral_cdf([theta](double u, double v) { return Gumbel::cdf(u, v, theta); });
            });
        return table;
    }
    if (nu != 4.0) throw std::invalid_argument("the t copula's Spearman table is built for nu = 4 only");
    static const auto t4 = student_t_spearman_table(4.0);
    return t4;
}

}  // namespace detail

inline double spearman_from_param(Family family, double param, double nu = 4.0) {
    if (family == Family::Gaussian) {
        return std::clamp(6.0 / std::numbers::pi * std::asin(param / 2.0), -1.0, 1.0);
    }
    return detail::spearman_table(family, nu).spearman(param);
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

// Match Spearman rho using the closed form or the family's calibration table.
inline double param_from_spearman(Family family, double target, double nu = 4.0) {
    if (family == Family::Gaussian) {
        const double r = 2.0 * std::sin(std::numbers::pi * std::clamp(target, -1.0, 1.0) / 6.0);
        return std::clamp(r, Gaussian::kMinParam, Gaussian::kMaxParam);
    }
    return detail::spearman_table(family, nu).inverse(target);
}

}  // namespace copula
