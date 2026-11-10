#pragma once

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "../src/estimators/copula_family.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"

// Gaussian misspecification floor. For a true copula density c the KL-closest Gaussian copula has
// r* = E_c[Z1 Z2] (Z = Phi^-1 of the margins), and KL(c || c_G(r*)) = E_c[log c] + 0.5 log(1 - r*^2).
// Both are integrated by the trapezoid rule in normal-score space. ew_marginals.md §33.
namespace gfloor {

struct Floor { double r, kl, mass, ez2, rho_s; };

inline Floor integrate(const std::function<double(double, double)>& logc, int n = 401, double L = 7.0) {
    const double h = 2.0 * L / (n - 1);
    std::vector<double> z(n), u(n), wphi(n);
    for (int i = 0; i < n; ++i) {
        z[i] = -L + h * i;
        u[i] = 0.5 * std::erfc(-z[i] / std::sqrt(2.0));
        wphi[i] = h * (i == 0 || i == n - 1 ? 0.5 : 1.0) * std::exp(-0.5 * z[i] * z[i]) / std::sqrt(2.0 * M_PI);
    }
    double mass = 0, exy = 0, elog = 0, ex2 = 0, euv = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double l = logc(u[i], u[j]);
            if (!std::isfinite(l)) continue;
            const double f = std::exp(l) * wphi[i] * wphi[j];
            mass += f; exy += f * z[i] * z[j]; elog += f * l; ex2 += f * z[i] * z[i]; euv += f * u[i] * u[j];
        }
    exy /= mass; elog /= mass; ex2 /= mass; euv /= mass;
    return {exy, elog + 0.5 * std::log(1.0 - exy * exy), mass, ex2, 12.0 * euv - 3.0};
}

// Truth = a mixture pi c_A(theta_A) + (1 - pi) c_B(theta_B) with both components at the rho_S of d
// (pi = 1 is a single family). Floors are tabulated on a 0.01 grid in d and interpolated linearly.
class Table {
public:
    Table(copula::Family a, copula::Family b, double nu) : a_(a), b_(b), nu_(nu) {}

    static double log_mix(copula::Family a, copula::Family b, double pi, double pa, double pb, double nu,
                          double u, double v) {
        const double la = copula::log_density(a, u, v, pa, nu);
        if (pi >= 1.0) return la;
        const double lb = copula::log_density(b, u, v, pb, nu);
        const double top = std::max(la, lb);
        return top + std::log(pi * std::exp(la - top) + (1.0 - pi) * std::exp(lb - top));
    }

    Floor at(double d, double pi) {
        const double x = d * 100.0, lo = std::floor(x + 1e-9);
        const Floor f0 = node(long(lo), pi);
        if (std::abs(x - lo) < 1e-9) return f0;
        const Floor f1 = node(long(lo) + 1, pi);
        const double w = x - lo;
        return {f0.r + w * (f1.r - f0.r), f0.kl + w * (f1.kl - f0.kl), 0, 0, f0.rho_s + w * (f1.rho_s - f0.rho_s)};
    }

private:
    Floor node(long k, double pi) {
        const auto key = std::make_pair(k, long(std::lround(pi * 1000)));
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
        const double ts = dgp::detail::target_spearman(k / 100.0);
        const double pa = copula::param_from_spearman(a_, ts, nu_), pb = copula::param_from_spearman(b_, ts, nu_);
        const auto a = a_, b = b_;
        const double nu = nu_;
        const Floor f = integrate([=](double u, double v) { return log_mix(a, b, pi, pa, pb, nu, u, v); });
        return cache_[key] = f;
    }

    copula::Family a_, b_;
    double nu_;
    std::map<std::pair<long, long>, Floor> cache_;
};

}  // namespace gfloor

// Clayton-Gumbel mixture DGP: each draw picks Clayton with probability pi_t, else Gumbel, both at the rho_S of
// d (tail_common.hpp's MixSampler rule); pi moves from pi0 to pi1 at the switch tick. rho_S is linear in C, so
// the mixture's rho_S is pi rho_C + (1 - pi) rho_G = the common target.
class MixSwitchCopula : public dgp::Copula {
public:
    MixSwitchCopula(double pi0, double pi1, std::size_t sw) : pi0_(pi0), pi1_(pi1), sw_(sw) {}
    double pi_at(std::size_t t) const { return t < sw_ ? pi0_ : pi1_; }
    dgp::CopulaDraw sample(double d, std::mt19937& rng) override {
        const double pi = pi_at(t_++);
        auto dr = unif_(rng) < pi ? cl_.sample(d, rng) : gu_.sample(d, rng);
        dr.rho_s = pi * copula::spearman_from_param(copula::Family::Clayton,
                            copula::param_from_spearman(copula::Family::Clayton, dgp::detail::target_spearman(d)))
                 + (1 - pi) * copula::spearman_from_param(copula::Family::Gumbel,
                            copula::param_from_spearman(copula::Family::Gumbel, dgp::detail::target_spearman(d)));
        return dr;
    }
    const char* name() const override { return "claygum_mix"; }

private:
    double pi0_, pi1_;
    std::size_t sw_, t_ = 0;
    dgp::ClaytonCopula cl_;
    dgp::GumbelCopula gu_;
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
};
