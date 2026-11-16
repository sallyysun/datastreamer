#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>

#include "../../estimators/copula_family.hpp"
#include "../../estimators/special_functions.hpp"

// Copula families for the synthetic DGP -- the generator counterpart to
// estimators/copula_family.hpp. Every family targets the same Spearman path
// rho_S = (6/pi) asin(d/2) and differs only in tail shape, so misspecification
// is isolated from the dependence trajectory. Notes: archive/notes/studies/roadmap.md.
namespace dgp {

struct CopulaDraw {
    double z1;
    double z2;
    double rho_s;
};

using copula::param_from_spearman;

struct Copula {
    virtual ~Copula() = default;
    virtual CopulaDraw sample(double d, std::mt19937& rng) = 0;
    virtual const char* name() const = 0;
};

namespace detail {
inline double target_spearman(double d) {
    const double rho = std::clamp(d, -0.999, 0.999);
    return 6.0 / std::numbers::pi * std::asin(rho / 2.0);
}
inline double to_normal_score(double u) {
    return special::normal_ppf(std::clamp(u, 1e-9, 1.0 - 1e-9));
}
}  // namespace detail

class GaussianCopula : public Copula {
public:
    CopulaDraw sample(double d, std::mt19937& rng) override {
        const double rho = std::clamp(d, -0.999, 0.999);
        const double z1 = normal_(rng);
        const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng);
        return {z1, z2, detail::target_spearman(d)};
    }
    const char* name() const override { return "gaussian"; }

private:
    std::normal_distribution<double> normal_{0.0, 1.0};
};

class ClaytonCopula : public Copula {
public:
    CopulaDraw sample(double d, std::mt19937& rng) override {
        const double target = detail::target_spearman(d);
        const double th = param_from_spearman(copula::Family::Clayton, target);
        const double u1 = unif_(rng);
        const double w = unif_(rng);
        const double u2 = std::pow(
            std::pow(u1, -th) * (std::pow(w, -th / (1.0 + th)) - 1.0) + 1.0, -1.0 / th);
        const double achieved = copula::spearman_from_param(copula::Family::Clayton, th);
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), achieved};
    }
    const char* name() const override { return "clayton"; }

private:
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
};

class GumbelCopula : public Copula {
public:
    CopulaDraw sample(double d, std::mt19937& rng) override {
        const double target = detail::target_spearman(d);
        const double th = param_from_spearman(copula::Family::Gumbel, target);
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
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), achieved};
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
    CopulaDraw sample(double d, std::mt19937& rng) override {
        const double target = detail::target_spearman(d);
        const double rho = param_from_spearman(copula::Family::StudentT, target, nu_);
        const double z1 = normal_(rng);
        const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng);
        const double s = std::sqrt(chi2_(rng) / nu_);
        const double u1 = special::t_cdf(z1 / s, nu_);
        const double u2 = special::t_cdf(z2 / s, nu_);
        const double achieved = copula::spearman_from_param(copula::Family::StudentT, rho, nu_);
        return {detail::to_normal_score(u1), detail::to_normal_score(u2), achieved};
    }
    const char* name() const override { return "student_t"; }

private:
    double nu_;
    std::normal_distribution<double> normal_{0.0, 1.0};
    std::gamma_distribution<double> chi2_{nu_ / 2.0, 2.0};
};

// Two families, one Spearman path, a switch part-way through. Because every
// family here already targets the SAME rho_S for a given knob d, holding d
// constant across the switch changes the copula's SHAPE -- its tail dependence --
// while leaving rho_S flat. That is distinction 1 made into an experiment: a
// change no rho-monitoring detector can see, by construction.
class SwitchingCopula : public Copula {
public:
    SwitchingCopula(std::unique_ptr<Copula> before, std::unique_ptr<Copula> after,
                    std::size_t switch_tick, const char* name)
        : before_(std::move(before)), after_(std::move(after)),
          switch_tick_(switch_tick), name_(name) {}

    CopulaDraw sample(double d, std::mt19937& rng) override {
        return (t_++ < switch_tick_) ? before_->sample(d, rng) : after_->sample(d, rng);
    }
    const char* name() const override { return name_; }

private:
    std::unique_ptr<Copula> before_, after_;
    std::size_t switch_tick_;
    const char* name_;
    std::size_t t_ = 0;
};

inline std::unique_ptr<Copula> make_copula(const std::string& name, double nu = 4.0) {
    // Gaussian -> t at matched Spearman rho, switching at N/2 as `jump` does.
    if (name == "gauss2t") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<GaussianCopula>(), std::make_unique<StudentTCopula>(nu),
            2500, "gauss2t");
    }
    // Clayton -> Gumbel at matched Spearman rho: lower-tail dependence swapped for upper.
    if (name == "clayton2gumbel") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<ClaytonCopula>(), std::make_unique<GumbelCopula>(), 2500, "clayton2gumbel");
    }
    // Gaussian -> Clayton at matched Spearman rho: lower-tail dependence appears.
    if (name == "gauss2clayton") {
        return std::make_unique<SwitchingCopula>(
            std::make_unique<GaussianCopula>(), std::make_unique<ClaytonCopula>(), 2500, "gauss2clayton");
    }
    if (name == "clayton") return std::make_unique<ClaytonCopula>();
    if (name == "gumbel")  return std::make_unique<GumbelCopula>();
    if (name == "t" || name == "student_t") return std::make_unique<StudentTCopula>(nu);
    return std::make_unique<GaussianCopula>();
}

}  // namespace dgp
