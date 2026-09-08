#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "../copula_family.hpp"
#include "../special_functions.hpp"
#include "estimator.hpp"
#include "marginal_rank_pit.hpp"

// Online MLE for the Clayton / Gumbel / Student-t copulas, fitted in an
// unconstrained coordinate phi by an empirical-Fisher natural-gradient step.
// Derivation: notes/studies/roadmap.md.
namespace copula {

class CopulaFit {
public:
    CopulaFit(Family family, double nu, double lambda, double vff_rate = 0.0)
        : family_(family), nu_(nu), lambda_(lambda), lambda0_(lambda), vff_rate_(vff_rate) {
        reset();
    }

    void reset() {
        phi_ = init_phi();
        info_ = kPriorInfo;
    }

    void update_uv(double u, double v) {
        const double p = param();
        double score_p = 0.0, dpar_dphi = 0.0;
        switch (family_) {
            case Family::Clayton:
                score_p = Clayton::score(u, v, p);
                dpar_dphi = p;
                break;
            case Family::Gumbel:
                score_p = Gumbel::score(u, v, p);
                dpar_dphi = p - 1.0;
                break;
            case Family::StudentT:
                score_p = StudentT::score(u, v, p, nu_);
                dpar_dphi = 1.0 - p * p;
                break;
            case Family::Gaussian:
            default:
                score_p = Gaussian::score(u, v, p);
                dpar_dphi = 1.0 - p * p;
                break;
        }
        const double score_phi = score_p * dpar_dphi;
        if (!std::isfinite(score_phi)) return;

        if (vff_rate_ > 0.0 && lambda0_ > 0.0) {
            // E[score] = 0 when theta is right. A systematically one-signed score
            // means theta lags, so shorten memory. lambda_ is RETENTION, so
            // shortening means lowering it.
            score_mean_ += vff_rate_ * (score_phi - score_mean_);
            score_var_ += vff_rate_ * (score_phi * score_phi - score_var_);
            const double drift = (score_mean_ * score_mean_) / std::max(score_var_, 1e-12);
            lambda_ = std::clamp(lambda0_ * std::pow(1.0 + drift, -10.0 * vff_rate_),
                                 0.90, 0.9999);
        }
        const double decay = (lambda_ > 0.0) ? lambda_ : 1.0;
        info_ = decay * info_ + score_phi * score_phi;
        double step = score_phi / info_;
        step = std::clamp(step, -kMaxStep, kMaxStep);
        phi_ += step;
    }

    double logpdf_uv(double u, double v) const {
        const double p = param();
        switch (family_) {
            case Family::Clayton:  return Clayton::log_density(u, v, p);
            case Family::Gumbel:   return Gumbel::log_density(u, v, p);
            case Family::StudentT: return StudentT::log_density(u, v, p, nu_);
            case Family::Gaussian:
            default:               return Gaussian::log_density(u, v, p);
        }
    }

    double param() const {
        switch (family_) {
            case Family::Clayton:
                return std::clamp(std::exp(phi_), Clayton::kMinParam, Clayton::kMaxParam);
            case Family::Gumbel:
                return std::clamp(1.0 + std::exp(phi_), Gumbel::kMinParam, Gumbel::kMaxParam);
            case Family::StudentT:
                return std::clamp(std::tanh(phi_), StudentT::kMinParam, StudentT::kMaxParam);
            case Family::Gaussian:
            default:
                return std::clamp(std::tanh(phi_), Gaussian::kMinParam, Gaussian::kMaxParam);
        }
    }

    double spearman() const { return spearman_from_param(family_, param(), nu_); }
    Family family() const { return family_; }

private:
    static constexpr double kPriorInfo = 1.0;
    static constexpr double kMaxStep = 0.5;

    double init_phi() const {
        switch (family_) {
            case Family::Clayton:  return std::log(0.5);
            case Family::Gumbel:   return std::log(0.5);
            case Family::StudentT: return 0.0;
            case Family::Gaussian:
            default:               return 0.0;
        }
    }

    Family family_;
    double nu_;
    double lambda_;
    double lambda0_ = 0.0;
    double vff_rate_ = 0.0;
    double score_mean_ = 0.0;
    double score_var_ = 1.0;
    double phi_ = 0.0;
    double info_ = kPriorInfo;
};

}  // namespace copula

class ParametricCopulaMLE : public DependenceEstimator {
public:
    struct Config {
        copula::Family family = copula::Family::Clayton;
        double nu = 4.0;
        double lambda = 0.0;
        double vff_rate = 0.0;
        // Rank-PIT marginals instead of a Gaussian CDF of the standardized value.
        // pseudo() is exact only for Gaussian margins; ranks are exact for any
        // continuous margin. Section 9.
        std::size_t rank_pit_window = 0;   // 0 keeps pseudo()
        std::size_t burn_in = 50;
    };

    ParametricCopulaMLE(Config config, const char* name)
        : config_(config), name_(name), fit_(config.family, config.nu, config.lambda, config.vff_rate) {}

    void update(double x, double y) override {
        ++n_;
        update_marginals(x, y);
        if (n_ < 2) return;
        double u, v;
        if (config_.rank_pit_window > 0) {
            u = pit_x_.transform(x);
            v = pit_y_.transform(y);
            if (pit_x_.size() < 30) return;   // no usable rank scale yet
        } else {
            u = pseudo(x, mean_x_, var_x_);
            v = pseudo(y, mean_y_, var_y_);
        }
        fit_.update_uv(u, v);
    }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0;
        return fit_.spearman();
    }

    void reset() override {
        n_ = 0;
        fit_.reset();
        mean_x_ = mean_y_ = 0.0;
        m2_x_ = m2_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
    }

    std::size_t store_bytes() const override { return sizeof(*this); }
    const char* name() const override { return name_; }

private:
    static constexpr double kEps = 1e-9;

    static double pseudo(double value, double mean, double var) {
        const double z = (value - mean) / std::sqrt(std::max(var, 1e-12));
        return std::clamp(special::normal_cdf(z), kEps, 1.0 - kEps);
    }

    void update_marginals(double x, double y) {
        const double n = static_cast<double>(n_);
        const double dx = x - mean_x_;
        const double dy = y - mean_y_;
        mean_x_ += dx / n;
        mean_y_ += dy / n;
        m2_x_ += dx * (x - mean_x_);
        m2_y_ += dy * (y - mean_y_);
        if (n_ > 1) {
            var_x_ = m2_x_ / (n - 1.0);
            var_y_ = m2_y_ / (n - 1.0);
        }
    }

    Config config_;
    // Declared after config_: a member initializer reading config_ from above it
    // would run before config_ exists.
    RankPit pit_x_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    RankPit pit_y_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    const char* name_;
    copula::CopulaFit fit_;
    std::size_t n_ = 0;

    double mean_x_ = 0.0, mean_y_ = 0.0;
    double m2_x_ = 0.0, m2_y_ = 0.0;
    double var_x_ = 1.0, var_y_ = 1.0;
};
