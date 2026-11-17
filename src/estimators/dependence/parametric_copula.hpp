#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "../copula_family.hpp"
#include "../special_functions.hpp"

// CopulaFit: online MLE for one copula family (Gaussian / Clayton / Gumbel / Student-t), fitted in an
// unconstrained coordinate phi by an expected-Fisher step. Used by the 6.4 family comparisons.
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
