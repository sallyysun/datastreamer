#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>

#include "dependence.hpp"
#include "study_config.hpp"
#include "../../copula_family.hpp"
#include "../../special_functions.hpp"

// Sample paired normal scores under the chosen copula family.
namespace dgp {

// target_spearman: the scenario's Spearman rho at this step; true_spearman: the Spearman rho the sampled parameter achieves.
struct CopulaDraw {
    double x, y;
    double target_spearman;
    double true_spearman;
};

struct Copula {
    virtual ~Copula() = default;
    virtual CopulaDraw sample(double target, std::mt19937& rng) = 0;
    virtual const char* name() const = 0;
};

namespace detail {
inline double to_normal_score(double u) {
    return special::normal_ppf(std::clamp(u, 1e-9, 1.0 - 1e-9));
}
}  // namespace detail

class GaussianCopula : public Copula {
public:
    CopulaDraw sample(double target, std::mt19937& rng) override {
        const double rho = copula::param_from_spearman(copula::Family::Gaussian, target);
        const double z1 = normal_(rng);
        const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng);
        return {z1, z2, target, copula::spearman_from_param(copula::Family::Gaussian, rho)};
    }
    const char* name() const override { return "gaussian"; }

private:
    std::normal_distribution<double> normal_{0.0, 1.0};
};

class ClaytonCopula : public Copula {
public:
    CopulaDraw sample(double target, std::mt19937& rng) override {
        const double th = copula::param_from_spearman(copula::Family::Clayton, target);
        const double u1 = unif_(rng);
        const double w = unif_(rng);
        const double u2 = std::pow(
            std::pow(u1, -th) * (std::pow(w, -th / (1.0 + th)) - 1.0) + 1.0, -1.0 / th);
        const double achieved = copula::spearman_from_param(copula::Family::Clayton, th);
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), target, achieved};
    }
    const char* name() const override { return "clayton"; }

private:
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
};

class GumbelCopula : public Copula {
public:
    CopulaDraw sample(double target, std::mt19937& rng) override {
        const double th = copula::param_from_spearman(copula::Family::Gumbel, target);
        double u1, u2;
        if (th <= 1.0 + 1e-9) {
            u1 = unif_(rng);
            u2 = unif_(rng);
        } else {
            const double m = positive_stable(1.0 / th, rng);
            const double e1 = exp_(rng), e2 = exp_(rng);
            u1 = std::exp(-std::pow(e1 / m, 1.0 / th));
            u2 = std::exp(-std::pow(e2 / m, 1.0 / th));
        }
        const double achieved = copula::spearman_from_param(copula::Family::Gumbel, th);
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), target, achieved};
    }
    const char* name() const override { return "gumbel"; }

private:
    double positive_stable(double alpha, std::mt19937& rng) {
        const double u = std::uniform_real_distribution<double>(1e-12, std::numbers::pi)(rng);
        const double w = exp_(rng);
        const double a = std::sin(alpha * u) / std::pow(std::sin(u), 1.0 / alpha);
        const double b = std::pow(std::sin((1.0 - alpha) * u) / w, (1.0 - alpha) / alpha);
        return a * b;
    }

    std::uniform_real_distribution<double> unif_{0.0, 1.0};
    std::exponential_distribution<double> exp_{1.0};
};

class StudentTCopula : public Copula {
public:
    explicit StudentTCopula(double nu) : nu_(nu) {}
    CopulaDraw sample(double target, std::mt19937& rng) override {
        const double rho = copula::param_from_spearman(copula::Family::StudentT, target, nu_);
        const double z1 = normal_(rng);
        const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng);
        const double s = std::sqrt(chi2_(rng) / nu_);
        const double u1 = special::t_cdf(z1 / s, nu_);
        const double u2 = special::t_cdf(z2 / s, nu_);
        const double achieved = copula::spearman_from_param(copula::Family::StudentT, rho, nu_);
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), target, achieved};
    }
    const char* name() const override { return "student_t"; }

private:
    double nu_;
    std::normal_distribution<double> normal_{0.0, 1.0};
    std::gamma_distribution<double> chi2_{nu_ / 2.0, 2.0};
};

// Switch copula shape at matched target Spearman rho.
class SwitchingCopula : public Copula {
public:
    SwitchingCopula(std::unique_ptr<Copula> before, std::unique_ptr<Copula> after,
                    std::size_t switch_tick, const char* name)
        : before_(std::move(before)), after_(std::move(after)),
          switch_tick_(switch_tick), name_(name) {}

    CopulaDraw sample(double target, std::mt19937& rng) override {
        return (t_++ < switch_tick_) ? before_->sample(target, rng) : after_->sample(target, rng);
    }
    const char* name() const override { return name_; }

private:
    std::unique_ptr<Copula> before_, after_;
    std::size_t switch_tick_;
    const char* name_;
    std::size_t t_ = 0;
};

inline std::unique_ptr<Copula> make_copula(const std::string& name, double nu = 4.0) {
    if (name == "gauss2t") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<GaussianCopula>(), std::make_unique<StudentTCopula>(nu),
            thesis::kChange, "gauss2t");
    }
    if (name == "clayton2gumbel") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<ClaytonCopula>(), std::make_unique<GumbelCopula>(), thesis::kChange, "clayton2gumbel");
    }
    if (name == "gauss2clayton") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<GaussianCopula>(), std::make_unique<ClaytonCopula>(), thesis::kChange, "gauss2clayton");
    }
    switch (copula::family_of(name)) {
        case copula::Family::Clayton: return std::make_unique<ClaytonCopula>();
        case copula::Family::Gumbel: return std::make_unique<GumbelCopula>();
        case copula::Family::StudentT: return std::make_unique<StudentTCopula>(nu);
        default: return std::make_unique<GaussianCopula>();
    }
}

}  // namespace dgp

// Draw paired normal scores from a seeded copula whose Spearman rho follows a dependence schedule.
class CopulaSource {
public:
    CopulaSource(Dependence& dependence, dgp::Copula& copula, unsigned seed)
        : dependence_(dependence), copula_(copula), rng_(seed) {}

    dgp::CopulaDraw next() { return draw(dependence_, copula_, rng_); }

    static dgp::CopulaDraw draw(Dependence& dependence, dgp::Copula& sampler, std::mt19937& rng) {
        return sampler.sample(std::clamp(dependence.next(), -1.0, 1.0), rng);
    }

private:
    Dependence& dependence_;
    dgp::Copula& copula_;
    std::mt19937 rng_;
};
