#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "../copula_family.hpp"
#include "../special_functions.hpp"
#include "estimator.hpp"
#include "marginal_rank_pit.hpp"
#include "predictive_density.hpp"

// Online MLE for the Clayton / Gumbel / Student-t copulas, fitted in an
// unconstrained coordinate phi by an empirical-Fisher natural-gradient step.
// Derivation: archive/notes/studies/roadmap.md.
namespace copula {

class CopulaFit {
public:
    CopulaFit(Family family, double nu, double lambda, double vff_rate = 0.0,
              bool expected_info = false, double pr_rate = 0.0, bool damped = false)
        : family_(family), nu_(nu), lambda_(lambda), lambda0_(lambda), vff_rate_(vff_rate),
          expected_info_(expected_info), pr_rate_(pr_rate), damped_(damped) {
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

        // (1) Expected rather than observed information (thesis section 3.5.4,
        // H_t = H_{t-1} + I(theta_{t-1})). score_phi^2 is a one-sample curvature
        // estimate and very noisy; expected_info_phi is closed form for the
        // Gaussian and tabulated for the others.
        const double contribution = expected_info_
            ? expected_info_phi(family_, param(), nu_)
            : score_phi * score_phi;
        info_ = decay * info_ + contribution;

        double step = score_phi / info_;

        // (3) Smooth saturation instead of a hard clamp. Same job, differentiable,
        // and it does not need re-tuning per family.
        step = damped_ ? kMaxStep * step / (kMaxStep + std::abs(step))
                       : std::clamp(step, -kMaxStep, kMaxStep);
        phi_ += step;

        // (2) Polyak-Ruppert: report an average of the iterates, not the last one.
        // A plain running mean assumes convergence to a fixed point, which a
        // forgetting estimator is not doing, so this is an EWMA slower than the
        // estimator itself.
        if (pr_rate_ > 0.0) {
            if (!pr_started_) { phi_avg_ = phi_; pr_started_ = true; }
            else              { phi_avg_ += pr_rate_ * (phi_ - phi_avg_); }
        }
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

    double param() const { return param_from_phi(phi_); }

    double param_from_phi(double phi) const {
        switch (family_) {
            case Family::Clayton:
                return std::clamp(std::exp(phi), Clayton::kMinParam, Clayton::kMaxParam);
            case Family::Gumbel:
                return std::clamp(1.0 + std::exp(phi), Gumbel::kMinParam, Gumbel::kMaxParam);
            case Family::StudentT:
                return std::clamp(std::tanh(phi), StudentT::kMinParam, StudentT::kMaxParam);
            case Family::Gaussian:
            default:
                return std::clamp(std::tanh(phi), Gaussian::kMinParam, Gaussian::kMaxParam);
        }
    }

    double spearman() const {
        const double phi = (pr_rate_ > 0.0 && pr_started_) ? phi_avg_ : phi_;
        return spearman_from_param(family_, param_from_phi(phi), nu_);
    }
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
    bool expected_info_ = false;
    double pr_rate_ = 0.0;
    bool damped_ = false;
    double phi_avg_ = 0.0;
    bool pr_started_ = false;
    double lambda_;
    double lambda0_ = 0.0;
    double vff_rate_ = 0.0;
    double score_mean_ = 0.0;
    double score_var_ = 1.0;
    double phi_ = 0.0;
    double info_ = kPriorInfo;
};

}  // namespace copula

class ParametricCopulaMLE : public DependenceEstimator, public PredictiveDensity {
public:
    struct Config {
        copula::Family family = copula::Family::Clayton;
        double nu = 4.0;
        double lambda = 0.0;
        double vff_rate = 0.0;
        bool expected_info = false;   // expected Fisher info, thesis section 3.5.4
        double pr_rate = 0.0;         // Polyak-Ruppert averaging rate; 0 disables
        bool damped = false;          // smooth step saturation instead of a clamp
        // Rank-PIT marginals instead of a Gaussian CDF of the standardized value.
        // pseudo() is exact only for Gaussian margins; ranks are exact for any
        // continuous margin. Section 9.
        std::size_t rank_pit_window = 0;   // 0 keeps pseudo()
        std::size_t burn_in = 50;
        // Inputs are already normal scores z = probit(u) of shared pseudo-observations (the bank
        // protocol): u = Phi(z), no standardisation.
        bool normal_score_inputs = false;
    };

    ParametricCopulaMLE(Config config, const char* name)
        : config_(config), name_(name), fit_(config.family, config.nu, config.lambda, config.vff_rate,
             config.expected_info, config.pr_rate, config.damped) {}

    void update(double x, double y) override {
        ++n_;
        update_marginals(x, y);
        if (n_ < 2 && !config_.normal_score_inputs) return;
        double u, v;
        if (config_.normal_score_inputs) {
            u = special::normal_cdf(x);
            v = special::normal_cdf(y);
        } else if (config_.rank_pit_window > 0) {
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
    bool ready() const override { return n_ >= config_.burn_in + 1 && n_ >= 2; }

    void reset() override {
        n_ = 0;
        fit_.reset();
        mean_x_ = mean_y_ = 0.0;
        m2_x_ = m2_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
    }

    // Fitted copula density at the component's own pseudo-observations p = Phi((z - mean)/sd) of the
    // inputs z = probit(u), times the Jacobian dp/du, so it is a density of (u, v).
    double log_predictive(double u, double v) const override {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (config_.rank_pit_window > 0) return kNaN;
        if (!(u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0)) return kNaN;
        if (n_ < config_.burn_in + 1) return 0.0;
        const auto side = [](double w, double mean, double var, double& p) {
            const double z = RankPit::probit(w), sd = std::sqrt(std::max(var, 1e-12)), s = (z - mean) / sd;
            p = std::clamp(special::normal_cdf(s), kEps, 1.0 - kEps);
            return 0.5 * (z * z - s * s) - std::log(sd);   // log dp/du = log phi(s) - log sd - log phi(z)
        };
        double pu = 0.0, pv = 0.0;
        const double jac = side(u, mean_x_, var_x_, pu) + side(v, mean_y_, var_y_, pv);
        return fit_.logpdf_uv(pu, pv) + jac;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + pit_x_.store_bytes() + pit_y_.store_bytes() - 2 * sizeof(RankPit);
    }
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
