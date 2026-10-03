#pragma once

// Fixed-parameter copula samplers and exact tail targets shared by the tail harnesses.
// See notes/studies/tail_dependence.md.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <string>
#include <future>
#include <utility>
#include <vector>

#include "../src/estimators/copula_family.hpp"
#include "../src/estimators/special_functions.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"

namespace tail {

constexpr double NU = 4.0;

struct Fixed {
    std::string family;
    double param = 0.0;  // Gaussian/t: correlation; Clayton/Gumbel: theta
};

inline Fixed fix(const std::string& f, double d) {
    const double target = 6.0 / std::numbers::pi * std::asin(std::clamp(d, -0.999, 0.999) / 2.0);
    if (f == "gaussian") return {f, std::clamp(d, -0.999, 0.999)};
    if (f == "clayton") return {f, dgp::param_from_spearman(copula::Family::Clayton, target)};
    if (f == "gumbel") return {f, dgp::param_from_spearman(copula::Family::Gumbel, target)};
    return {f, dgp::param_from_spearman(copula::Family::StudentT, target, NU)};
}

// Same formulas as copula.hpp's samplers, parameter held fixed. Returns (u, v) directly.
class Sampler {
public:
    Sampler(Fixed f, unsigned seed) : f_(std::move(f)), rng_(seed) {}
    std::pair<double, double> next() {
        const double p = f_.param;
        if (f_.family == "gaussian" || f_.family == "t") {
            const double z1 = normal_(rng_);
            const double z2 = p * z1 + std::sqrt(1.0 - p * p) * normal_(rng_);
            if (f_.family == "gaussian") return {special::normal_cdf(z1), special::normal_cdf(z2)};
            const double s = std::sqrt(chi2_(rng_) / NU);
            return {special::t_cdf(z1 / s, NU), special::t_cdf(z2 / s, NU)};
        }
        if (f_.family == "clayton") {
            const double u1 = unif_(rng_), w = unif_(rng_);
            return {u1, std::pow(std::pow(u1, -p) * (std::pow(w, -p / (1.0 + p)) - 1.0) + 1.0, -1.0 / p)};
        }
        const double m = positive_stable(1.0 / p);
        const double e1 = exp_(rng_), e2 = exp_(rng_);
        return {std::exp(-std::pow(e1 / m, 1.0 / p)), std::exp(-std::pow(e2 / m, 1.0 / p))};
    }

private:
    double positive_stable(double alpha) {
        const double u = std::uniform_real_distribution<double>(1e-12, std::numbers::pi)(rng_);
        const double w = exp_(rng_);
        const double a = std::sin(alpha * u) / std::pow(std::sin(u), 1.0 / alpha);
        const double b = std::pow(std::sin((1.0 - alpha) * u) / w, (1.0 - alpha) / alpha);
        return a * b;
    }
    Fixed f_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
    std::exponential_distribution<double> exp_{1.0};
    std::gamma_distribution<double> chi2_{NU / 2.0, 2.0};
};

struct Lambdas { double lower, upper; };

inline Lambdas exact_lambda(const Fixed& f) {
    if (f.family == "gaussian") return {0.0, 0.0};
    if (f.family == "clayton") return {std::pow(2.0, -1.0 / f.param), 0.0};
    if (f.family == "gumbel") return {0.0, 2.0 - std::pow(2.0, 1.0 / f.param)};
    const double l = 2.0 * special::t_cdf(-std::sqrt((NU + 1.0) * (1.0 - f.param) / (1.0 + f.param)), NU + 1.0);
    return {l, l};
}

// Closed-form C(a, a) where it exists; NAN otherwise (Monte Carlo is used instead).
inline double closed_C(const Fixed& f, double a) {
    if (f.family == "clayton") return std::pow(2.0 * std::pow(a, -f.param) - 1.0, -1.0 / f.param);
    if (f.family == "gumbel") return std::exp(-std::pow(2.0, 1.0 / f.param) * (-std::log(a)));
    return NAN;
}

// Monte Carlo corner probabilities P(U<=q,V<=q), P(U>1-q,V>1-q) per q, and rho_S = 12E[UV]-3.
struct Corners {
    double rho_s;
    std::vector<double> pL, pU;
};

inline Corners exact_corners(const Fixed& f, const std::vector<double>& qs, std::size_t n_total) {
    const unsigned threads = 8;
    std::vector<std::future<std::vector<double>>> jobs;
    for (unsigned k = 0; k < threads; ++k)
        jobs.push_back(std::async(std::launch::async, [&f, &qs, n_total, k] {
            Sampler s(f, 900001u + k);
            std::vector<double> acc(1 + 2 * qs.size(), 0.0);
            for (std::size_t i = 0; i < n_total / threads; ++i) {
                const auto [u, v] = s.next();
                acc[0] += u * v;
                for (std::size_t j = 0; j < qs.size(); ++j) {
                    acc[1 + j] += (u <= qs[j] && v <= qs[j]);
                    acc[1 + qs.size() + j] += (u > 1.0 - qs[j] && v > 1.0 - qs[j]);
                }
            }
            return acc;
        }));
    std::vector<double> acc(1 + 2 * qs.size(), 0.0);
    for (auto& j : jobs) { auto a = j.get(); for (std::size_t i = 0; i < a.size(); ++i) acc[i] += a[i]; }
    const double n = static_cast<double>(n_total / threads * threads);
    Corners c{12.0 * acc[0] / n - 3.0, {}, {}};
    for (std::size_t j = 0; j < qs.size(); ++j) {
        c.pL.push_back(acc[1 + j] / n);
        c.pU.push_back(acc[1 + qs.size() + j] / n);
    }
    return c;
}

}  // namespace tail
