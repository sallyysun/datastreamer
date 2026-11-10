#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "../copula_family.hpp"
#include "../special_functions.hpp"
#include "estimator.hpp"
#include "ew_marginal.hpp"
#include "marginal_rank_pit.hpp"
#include "parametric_copula.hpp"
#include "tail_estimator.hpp"

// Family bank: thesis Algorithm 1 with arms over copula shapes x memories. Each arm is a forgetting
// online MPL (CopulaFit, expected Fisher scoring) on the shared rank PIT, scored by its own
// pre-update log copula density, so the bank's predictive density is the weighted mixture of
// proper copula densities on one scale and the section 5.3.4 log-loss bound holds at eta = 1.
// Outputs: rho_S of the mixture (sum w_k rho_S,k), and tails linear in C (lambda_L, lambda_U,
// chi_L(q), chi_U(q)) as the same weighted sums. Clayton and Gumbel also come rotated by 90 degrees
// (c(1-u, v)) and 270 degrees (c(u, 1-v)) so that negative dependence has an arm; the two are
// transposes of each other, not equal, since neither family is radially symmetric.
// notes/studies/thesis_bank_family.md.
namespace copula {

enum class Shape { Gaussian, StudentT, Clayton, Clayton90, Gumbel, Gumbel90, Clayton270, Gumbel270 };
constexpr int kShapes = 8;

inline Family base_family(Shape s) {
    switch (s) {
        case Shape::StudentT: return Family::StudentT;
        case Shape::Clayton: case Shape::Clayton90: case Shape::Clayton270: return Family::Clayton;
        case Shape::Gumbel: case Shape::Gumbel90: case Shape::Gumbel270: return Family::Gumbel;
        case Shape::Gaussian: default: return Family::Gaussian;
    }
}
inline bool rot90(Shape s) { return s == Shape::Clayton90 || s == Shape::Gumbel90; }
inline bool rot270(Shape s) { return s == Shape::Clayton270 || s == Shape::Gumbel270; }
inline bool rotated(Shape s) { return rot90(s) || rot270(s); }
// The base copula's coordinates of a pair: 90 degrees is (1-u, v), 270 degrees is (u, 1-v).
inline std::pair<double, double> base_uv(Shape s, double u, double v) {
    return rot90(s) ? std::pair{1.0 - u, v} : rot270(s) ? std::pair{u, 1.0 - v} : std::pair{u, v};
}

struct Tails { double lam_L, lam_U, chi_L, chi_U; };

namespace detail {

// C(a, b) of the Gaussian / t copula as int_0^a P(V <= b | U = w) dw (Gauss-Legendre in w).
inline double elliptical_cdf(Family f, double rho, double nu, double a, double b) {
    static const GaussLegendre gl(64);
    const double s = std::sqrt(std::max(1.0 - rho * rho, 1e-12));
    const double y = f == Family::StudentT ? special::t_ppf(b, nu) : special::normal_ppf(b);
    double total = 0.0;
    for (std::size_t i = 0; i < gl.x.size(); ++i) {
        const double w = a * gl.x[i];
        if (f == Family::StudentT) {
            const double x = special::t_ppf(w, nu);
            const double sc = std::sqrt((nu + x * x) / (nu + 1.0)) * s;
            total += gl.w[i] * special::t_cdf((y - rho * x) / sc, nu + 1.0);
        } else {
            const double x = special::normal_ppf(w);
            total += gl.w[i] * special::normal_cdf((y - rho * x) / s);
        }
    }
    return a * total;
}

}  // namespace detail

inline double base_cdf(Family f, double par, double nu, double a, double b) {
    switch (f) {
        case Family::Clayton: return Clayton::cdf(a, b, par);
        case Family::Gumbel: return Gumbel::cdf(a, b, par);
        default: return detail::elliptical_cdf(f, par, nu, a, b);
    }
}

// Exact tails of one shape at parameter par. chi_L(q) = C(q,q)/q, chi_U(q) = P(U>1-q, V>1-q)/q.
// Rotations put the base copula's tail in an off-diagonal corner, so lambda_L = lambda_U = 0:
// C90(a, b) = b - C(1-a, b), C270(a, b) = a - C(a, 1-b).
inline Tails exact_tails(Shape s, double par, double nu, double q) {
    const Family f = base_family(s);
    if (rot90(s))
        return {0.0, 0.0, (q - base_cdf(f, par, nu, 1.0 - q, q)) / q, (q - base_cdf(f, par, nu, q, 1.0 - q)) / q};
    if (rot270(s))
        return {0.0, 0.0, (q - base_cdf(f, par, nu, q, 1.0 - q)) / q, (q - base_cdf(f, par, nu, 1.0 - q, q)) / q};
    switch (f) {
        case Family::Clayton:
            return {std::pow(2.0, -1.0 / par), 0.0, Clayton::cdf(q, q, par) / q,
                    (2.0 * q - 1.0 + Clayton::cdf(1.0 - q, 1.0 - q, par)) / q};
        case Family::Gumbel:
            return {0.0, 2.0 - std::pow(2.0, 1.0 / par), Gumbel::cdf(q, q, par) / q,
                    (2.0 * q - 1.0 + Gumbel::cdf(1.0 - q, 1.0 - q, par)) / q};
        case Family::StudentT: {
            const double l = 2.0 * special::t_cdf(-std::sqrt((nu + 1.0) * (1.0 - par) / (1.0 + par)), nu + 1.0);
            const double c = detail::elliptical_cdf(f, par, nu, q, q) / q;   // radially symmetric
            return {l, l, c, c};
        }
        default: {
            const double c = detail::elliptical_cdf(f, par, nu, q, q) / q;
            return {0.0, 0.0, c, c};
        }
    }
}

// chi_L(q) of the Gaussian / t copula tabulated in rho (exact_tails is a 64-point quadrature with
// quantile bisections; the bank needs it every tick).
inline double elliptical_chi(Family f, double rho, double nu, double q) {
    constexpr int kGrid = 400;
    static std::map<std::pair<int, std::pair<double, double>>, std::vector<double>> cache;
    auto& tab = cache[{static_cast<int>(f), {nu, q}}];
    if (tab.empty())
        for (int k = 0; k <= kGrid; ++k) {
            const double r = -0.999 + 1.998 * k / kGrid;
            tab.push_back(detail::elliptical_cdf(f, r, nu, q, q) / q);
        }
    const double x = (std::clamp(rho, -0.999, 0.999) + 0.999) / 1.998 * kGrid;
    const int k = std::min(static_cast<int>(x), kGrid - 1);
    return tab[k] + (x - k) * (tab[k + 1] - tab[k]);
}

}  // namespace copula

class FamilyBank : public DependenceEstimator, public TailEstimator {
public:
    struct Config {
        std::vector<copula::Shape> shapes{copula::Shape::Gaussian, copula::Shape::StudentT, copula::Shape::Clayton,
                                          copula::Shape::Clayton90, copula::Shape::Clayton270, copula::Shape::Gumbel,
                                          copula::Shape::Gumbel90, copula::Shape::Gumbel270};
        std::vector<int> memories{99, 399, 1999};   // retention lambda = 1 - 2/(M+1)
        double eta = 1.0;
        double delta = 2.1e-4;         // delta* = K m / ((K-1)(T-1)) at K = 24, m = 1, T = 4970
        double nu = 4.0;
        double marginal_neff = default_marginal_neff();   // shared EW marginal (ew_marginal.hpp)
        std::size_t warm_up = 30;
    };

    struct Arm {
        copula::Shape shape;
        int mem;
        copula::CopulaFit fit;
    };

    FamilyBank(Config config, const char* name) : config_(std::move(config)), name_(name),
        pit_x_(config_.marginal_neff), pit_y_(config_.marginal_neff) {
        for (auto s : config_.shapes)
            for (int m : config_.memories)
                arms_.push_back({s, m, copula::CopulaFit(copula::base_family(s), config_.nu, 1.0 - 2.0 / (m + 1.0),
                                                         0.0, true)});
        w_.assign(arms_.size(), 1.0 / static_cast<double>(arms_.size()));
        loss_.assign(arms_.size(), 0.0);
        lw_.assign(arms_.size(), 0.0);
    }

    void update(double x, double y) override {
        const std::size_t K = arms_.size();
        const bool scorable = pit_x_.size() >= config_.warm_up;
        const double u = pit_x_.transform(x);
        const double v = pit_y_.transform(y);
        if (!scorable) return;

        for (std::size_t k = 0; k < K; ++k) loss_[k] = -arm_log_density(k, u, v);
        // Guard: a non-finite loss is replaced by the worst finite loss this tick (as ForgettingBank).
        double worst = 0.0;
        for (double l : loss_) if (std::isfinite(l)) worst = std::max(worst, l);
        for (double& l : loss_) if (!std::isfinite(l)) l = worst;

        double top = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < K; ++k) {
            lw_[k] = (w_[k] > 0.0 ? std::log(w_[k]) : -std::numeric_limits<double>::infinity())
                   - config_.eta * loss_[k];
            top = std::max(top, lw_[k]);
        }
        double total = 0.0;
        for (std::size_t k = 0; k < K; ++k) total += (w_[k] = std::exp(lw_[k] - top));
        const double share = config_.delta / static_cast<double>(K);
        for (double& wk : w_) wk = (1.0 - config_.delta) * wk / total + share;

        for (auto& a : arms_) {
            const auto [ua, va] = copula::base_uv(a.shape, u, v);
            a.fit.update_uv(ua, va);
        }
        ++n_;
    }

    double estimate() const override {
        if (n_ == 0) return 0.0;
        double r = 0.0;
        for (std::size_t k = 0; k < arms_.size(); ++k) r += w_[k] * arm_rho(k);
        return std::clamp(r, -1.0, 1.0);
    }

    // Log density of arm k at (u, v) under its current fit, on the copula scale.
    double arm_log_density(std::size_t k, double u, double v) const {
        const auto [ua, va] = copula::base_uv(arms_[k].shape, u, v);
        return arms_[k].fit.logpdf_uv(ua, va);
    }

    // Log of the mixture copula density sum_k w_k c_k(u, v): the bank's predictive density.
    double log_density(double u, double v) const {
        double top = -std::numeric_limits<double>::infinity();
        std::vector<double> l(arms_.size());
        for (std::size_t k = 0; k < arms_.size(); ++k) {
            l[k] = std::log(w_[k]) + arm_log_density(k, u, v);
            if (!std::isfinite(l[k])) l[k] = -std::numeric_limits<double>::infinity();
            top = std::max(top, l[k]);
        }
        if (!std::isfinite(top)) return top;
        double s = 0.0;
        for (double x : l) s += std::exp(x - top);
        return top + std::log(s);
    }

    double arm_rho(std::size_t k) const {
        const double r = arms_[k].fit.spearman();
        return copula::rotated(arms_[k].shape) ? -r : r;
    }

    copula::Tails arm_tails(std::size_t k, double q) const {
        const auto& a = arms_[k];
        const double p = a.fit.param();
        const copula::Family f = copula::base_family(a.shape);
        if (f == copula::Family::Gaussian || f == copula::Family::StudentT) {
            const double c = copula::elliptical_chi(f, p, config_.nu, q);
            const double l = f == copula::Family::Gaussian ? 0.0
                : 2.0 * special::t_cdf(-std::sqrt((config_.nu + 1.0) * (1.0 - p) / (1.0 + p)), config_.nu + 1.0);
            return {l, l, c, c};
        }
        return copula::exact_tails(a.shape, p, config_.nu, q);
    }

    // Mixture tails: every one is linear in C, so the weighted sum of the arms' values.
    TailEstimate tails(double q) const override {
        TailEstimate t{0, 0, 0, 0};
        for (std::size_t k = 0; k < arms_.size(); ++k) {
            const auto a = arm_tails(k, q);
            t.lambda_L += w_[k] * a.lam_L;
            t.lambda_U += w_[k] * a.lam_U;
            t.chi_L += w_[k] * a.chi_L;
            t.chi_U += w_[k] * a.chi_U;
        }
        return t;
    }

    void reset() override {
        for (auto& a : arms_) a.fit.reset();
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(arms_.size()));
        std::fill(loss_.begin(), loss_.end(), 0.0);
        pit_x_.reset();
        pit_y_.reset();
        n_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + pit_x_.heap_bytes() + pit_y_.heap_bytes()
             + arms_.capacity() * sizeof(Arm) + (w_.capacity() + loss_.capacity() + lw_.capacity()) * sizeof(double);
    }

    const char* name() const override { return name_; }

    std::size_t size() const { return arms_.size(); }
    double weight(std::size_t k) const { return w_[k]; }
    const Arm& arm(std::size_t k) const { return arms_[k]; }
    const Config& config() const { return config_; }

private:
    Config config_;
    const char* name_;
    EwMarginal pit_x_, pit_y_;
    std::vector<Arm> arms_;
    std::vector<double> w_, loss_, lw_;
    std::size_t n_ = 0;
};
