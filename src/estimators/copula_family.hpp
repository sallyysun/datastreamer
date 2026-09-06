#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "objective.hpp"
#include "special_functions.hpp"

// Parametric copula families for the parametric branch of the estimator bank
// (companion to the Gaussian-copula MLE already in objective.hpp). Each family
// exposes, on pseudo-observations (u, v) in (0,1)^2:
//   * log_density(u, v, param)  -- the copula log-density log c(u,v; param);
//   * score(u, v, param)        -- d/dparam log c, the MLE score in the family's
//                                  NATURAL parameter (theta for the Archimedean
//                                  families, rho for the t copula). Verified
//                                  against central finite differences in the test.
// and every family shares one Spearman conversion:
//   * spearman_from_param(param) via  rho_S = 12 * E[U V] - 3
//                                    = 12 * integral u v c(u,v) du dv - 3,
//     evaluated by tensor Gauss-Legendre on the density. Using the density (not
//     the CDF) makes the same formula work for Clayton, Gumbel AND t, so every
//     estimator still reports Spearman's rho and stays comparable to the
//     nonparametric branch. See registry.hpp for the forgetting conventions.
//
// Scope / assumptions:
//   * Clayton and Gumbel are single-parameter and capture POSITIVE dependence
//     only (theta -> lower bound => independence). Under negatively-dependent
//     data the MLE collapses to rho_S ~ 0 -- that is correct misspecification
//     behaviour, and exactly what the GoF selector is meant to reveal.
//   * The t copula's degrees of freedom nu are a FIXED hyper-parameter (not
//     estimated online -- estimating nu from a stream is a separate problem);
//     only the correlation rho is fitted.
namespace copula {

enum class Family { Gaussian, Clayton, Gumbel, StudentT };

// --- Gauss-Legendre nodes on (0,1), generated once (Numerical Recipes gauleg) ---
struct GaussLegendre {
    std::vector<double> x;  // nodes in (0,1)
    std::vector<double> w;  // weights

    explicit GaussLegendre(int n) : x(n), w(n) {
        constexpr double kEps = 1.0e-14;
        const int m = (n + 1) / 2;
        for (int i = 0; i < m; ++i) {
            // Initial guess for the i-th root of the Legendre polynomial on [-1,1].
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
            // Map [-1,1] -> [0,1].
            x[i] = 0.5 * (1.0 - z);
            x[n - 1 - i] = 0.5 * (1.0 + z);
            const double weight = 1.0 / ((1.0 - z * z) * pp * pp);  // *2 for [-1,1], *0.5 for scale
            w[i] = weight;
            w[n - 1 - i] = weight;
        }
    }
};

// ----------------------------- Clayton -----------------------------
// c(u,v;theta) = (1+theta) (uv)^{-(1+theta)} (u^{-theta}+v^{-theta}-1)^{-(2+1/theta)},
// theta > 0. Lower-tail dependence.
struct Clayton {
    static constexpr double kMinParam = 1.0e-3;
    static constexpr double kMaxParam = 50.0;

    static double log_density(double u, double v, double theta) {
        const double S = std::pow(u, -theta) + std::pow(v, -theta) - 1.0;
        return std::log1p(theta)
               - (1.0 + theta) * (std::log(u) + std::log(v))
               - (2.0 + 1.0 / theta) * std::log(S);
    }

    // C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}. Used for the (smooth,
    // bounded) CDF form of the Spearman integral.
    static double cdf(double u, double v, double theta) {
        return std::pow(std::pow(u, -theta) + std::pow(v, -theta) - 1.0, -1.0 / theta);
    }

    // dl/dtheta = 1/(1+theta) - (log u + log v) + (1/theta^2) log S
    //             - (2 + 1/theta) * S'/S,   S' = -(u^{-theta} log u + v^{-theta} log v).
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

// ----------------------------- Gumbel ------------------------------
// x=-log u, y=-log v, w=x^theta+y^theta, A=w^{1/theta}, theta >= 1.
// c = C(u,v) (xy)^{theta-1}/(uv) w^{1/theta-2} (A+theta-1),  C=exp(-A).
// Upper-tail dependence.
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

    // C(u,v) = exp(-((-log u)^theta + (-log v)^theta)^{1/theta}).
    static double cdf(double u, double v, double theta) {
        const double x = -std::log(u), y = -std::log(v);
        return std::exp(-std::pow(std::pow(x, theta) + std::pow(y, theta), 1.0 / theta));
    }

    // Derivative w.r.t. theta with intermediates w' = x^theta log x + y^theta log y,
    // L = log w, A' = A [ w'/(theta w) - L/theta^2 ].
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

// ----------------------------- Gaussian ----------------------------
// Wraps GaussianCopulaObjective (normal scores) in the (u,v) interface the fit
// and selector use, so every family is scored uniformly. Natural parameter rho.
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

// ----------------------------- Student-t ---------------------------
// Normal-score analogue with t scores x = t_nu^{-1}(u), y = t_nu^{-1}(v).
// Correlation rho fitted; nu fixed. Symmetric tail dependence.
struct StudentT {
    static constexpr double kMinParam = -0.999;
    static constexpr double kMaxParam = 0.999;

    static double log_density(double u, double v, double rho, double nu) {
        const double x = special::t_ppf(u, nu);
        const double y = special::t_ppf(v, nu);
        const double d = 1.0 - rho * rho;
        const double Q = x * x - 2.0 * rho * x * y + y * y;
        // log bivariate-t joint density minus the two marginal t log-densities.
        const double log_joint =
            std::lgamma((nu + 2.0) / 2.0) - std::lgamma(nu / 2.0)
            - std::log(nu * std::numbers::pi) - 0.5 * std::log(d)
            - (nu + 2.0) / 2.0 * std::log1p(Q / (nu * d));
        const double log_mx = std::log(special::t_pdf(x, nu));
        const double log_my = std::log(special::t_pdf(y, nu));
        return log_joint - log_mx - log_my;
    }

    // dl/drho = rho/d - ((nu+2)/(nu d^2 g))(rho Q - x y d),  g = 1 + Q/(nu d).
    static double score(double u, double v, double rho, double nu) {
        const double x = special::t_ppf(u, nu);
        const double y = special::t_ppf(v, nu);
        const double d = 1.0 - rho * rho;
        const double Q = x * x - 2.0 * rho * x * y + y * y;
        const double g = 1.0 + Q / (nu * d);
        return rho / d - (nu + 2.0) / (nu * d * d * g) * (rho * Q - x * y * d);
    }
};

// --------- Spearman's rho from the family parameter via quadrature ----------
// rho_S = 12 * integral_0^1 integral_0^1 u v c(u,v; param) du dv - 3.
// Built once into a monotone lookup table per (family, param range) and linearly
// interpolated; rho_S(param) is smooth and monotone so this is accurate and O(1).
namespace detail {

// rho_S = 12 integral u v c(u,v) - 3, integrating the DENSITY. Fine for the
// smooth Gaussian / t densities; avoid it for the Archimedean families whose
// density has boundary singularities that bias low-order quadrature (use the CDF
// form below there instead).
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

// rho_S = 12 integral C(u,v) - 3, integrating the (smooth, bounded) CDF. The
// accurate form for the Archimedean families -- their CDF has no boundary
// singularity, so Gauss-Legendre converges fast.
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

// A monotone table param -> rho_S over [lo, hi] with a forward linear lookup.
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
    constexpr int kGrid = 96;  // table resolution; interpolation is linear
    switch (family) {
        case Family::Clayton: {
            static const detail::SpearmanTable table = [] {
                detail::SpearmanTable t;
                for (int k = 0; k <= kGrid; ++k) {
                    // Geometric spacing: rho_S changes fast for small theta.
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
                // Geometric spacing in (theta - 1): rho_S rises steeply and convexly
                // near the independence limit theta = 1, so a linear grid there would
                // make interpolation badly underestimate (cf. Clayton's geometric grid).
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
            // rho_S depends on nu, so cache the table keyed on nu (nu is fixed per
            // run in practice) -- rebuilding this quadrature per estimate() call
            // would be prohibitively slow given the t-quantile evaluations.
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
            // Closed form for the Gaussian copula.
            return std::clamp(6.0 / std::numbers::pi * std::asin(param / 2.0), -1.0, 1.0);
    }
}

}  // namespace copula
