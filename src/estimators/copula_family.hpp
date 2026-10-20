#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "objective.hpp"
#include "special_functions.hpp"

// Parametric copula families: log-density, score in the natural parameter, Spearman by Gauss-Legendre.
// Clayton/Gumbel are positive-dependence only; the t copula's nu is fixed.
// Derivations: notes/studies/roadmap.md.
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

    static double score(double u, double v, double theta) {
        const double a = std::pow(u, -theta);
        const double b = std::pow(v, -theta);
        const double S = a + b - 1.0;
        const double Sp = -(a * std::log(u) + b * std::log(v));
        return 1.0 / (1.0 + theta)
               - (std::log(u) + std::log(v))
               + std::log(S) / (theta * theta)
               - (2.0 + 1.0 / theta) * (Sp / S);
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

    static double score(double u, double v, double theta) {
        const double x = -std::log(u);
        const double y = -std::log(v);
        const double xt = std::pow(x, theta);
        const double yt = std::pow(y, theta);
        const double w = xt + yt;
        const double L = std::log(w);
        const double wp = xt * std::log(x) + yt * std::log(y);
        const double A = std::pow(w, 1.0 / theta);
        const double Ap = A * (wp / (theta * w) - L / (theta * theta));
        return -Ap
               + (std::log(x) + std::log(y))
               - L / (theta * theta)
               + (1.0 / theta - 2.0) * (wp / w)
               + (Ap + 1.0) / (A + theta - 1.0);
    }
};

struct Gaussian {
    static constexpr double kMinParam = -0.999;
    static constexpr double kMaxParam = 0.999;

    static double log_density(double u, double v, double rho) {
        return GaussianCopulaObjective::log_likelihood(
            special::normal_ppf(u), special::normal_ppf(v), rho);
    }
    static double score(double u, double v, double rho) {
        return GaussianCopulaObjective::score(
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

    static double score(double u, double v, double rho, double nu) {
        const double x = special::t_ppf(u, nu);
        const double y = special::t_ppf(v, nu);
        const double d = 1.0 - rho * rho;
        const double Q = x * x - 2.0 * rho * x * y + y * y;
        const double g = 1.0 + Q / (nu * d);
        return rho / d - (nu + 2.0) / (nu * d * d * g) * (rho * Q - x * y * d);
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

}

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
            static double cached_nu = -1.0;
            static detail::SpearmanTable table;
            if (nu != cached_nu) {
                table = detail::SpearmanTable{};
                for (int k = 0; k <= kGrid; ++k) {
                    const double r = StudentT::kMinParam
                        + (StudentT::kMaxParam - StudentT::kMinParam) * static_cast<double>(k) / kGrid;
                    table.param.push_back(r);
                    table.rho_s.push_back(detail::spearman_integral(
                        [r, nu](double u, double v) { return std::exp(StudentT::log_density(u, v, r, nu)); }));
                }
                cached_nu = nu;
            }
            return std::clamp(table.lookup(std::clamp(param, StudentT::kMinParam, StudentT::kMaxParam)), -1.0, 1.0);
        }
        case Family::Gaussian:
        default:
            return std::clamp(6.0 / std::numbers::pi * std::asin(param / 2.0), -1.0, 1.0);
    }
}

}
