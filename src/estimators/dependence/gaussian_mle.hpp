#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "estimator.hpp"
#include "../objective.hpp"

// Online Gaussian-copula MLE via recursive Fisher scoring (the parametric branch;
// roadmap: "online MLE / Fisher scoring"). Parameterized in theta = atanh(rho),
// which is unconstrained and makes the Fisher information in theta simply
// 1 + rho^2 (bounded, well-behaved near +-1).
//
// Marginals are standardized online with cumulative Welford moments: the marginal
// scale is stationary, so it is NOT forgotten -- only the dependence is. This
// decouples the two timescales and avoids the hermite_ew cold-start (where a
// forgotten variance took ~800 ticks to reach the right scale). For Gaussian
// marginals, standardization gives exactly the normal scores.
//
// The dependence update is a natural-gradient (Fisher-preconditioned) step whose
// learning rate is set by an accumulated information H:
//   H_t     = lambda * H_{t-1} + I_theta         (lambda = 1 => cumulative MLE)
//   theta_t = theta_{t-1} + s_theta / H_t
// lambda in (0,1) gives exponential forgetting (ESS ~ 1/(1-lambda)); lambda = 0
// (the default sentinel) means cumulative (recursive MLE, step ~ 1/n).
class GaussianCopulaMLE : public DependenceEstimator {
public:
    struct Config {
        double lambda = 0.0;       // forgetting for the dependence info; 0 => cumulative
        std::size_t burn_in = 50;  // samples before reporting an estimate
    };

    GaussianCopulaMLE(Config config, const char* name)
        : config_(config), name_(name) {}

    void update(double x, double y) override {
        ++n_;
        update_marginals(x, y);
        if (n_ < 2) return;

        const double zx = (x - mean_x_) / sd(var_x_);
        const double zy = (y - mean_y_) / sd(var_y_);

        const double rho = std::tanh(theta_);
        const double drho_dtheta = 1.0 - rho * rho;
        const double s_theta = GaussianCopulaObjective::score(zx, zy, rho) * drho_dtheta;
        const double i_theta = 1.0 + rho * rho;  // Fisher information in theta
        if (!std::isfinite(s_theta)) return;

        const double decay = (config_.lambda > 0.0) ? config_.lambda : 1.0;
        info_ = decay * info_ + i_theta;
        // score(.,.,rho) carries 1/(1-rho^2)^2 and drho_dtheta cancels only one
        // power of it, so theta is clamped away from the saturation of tanh: at
        // |rho| = 1 exactly the next step is inf, and opposite-signed infinities
        // then latch theta at NaN for the rest of the stream. Same guards, and
        // the same +-0.999 bound, as ParametricCopulaMLE.
        theta_ = std::clamp(theta_ + s_theta / info_, -kMaxTheta, kMaxTheta);
    }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0;
        const double rho = std::tanh(theta_);
        // Gaussian copula: Spearman's rho = (6/pi) asin(rho/2) -- same space as
        // the nonparametric estimators, so results are directly comparable.
        return std::clamp(6.0 / std::numbers::pi * std::asin(rho / 2.0), -1.0, 1.0);
    }

    void reset() override {
        n_ = 0;
        theta_ = 0.0;
        info_ = 0.0;
        mean_x_ = mean_y_ = 0.0;
        m2_x_ = m2_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
    }

    std::size_t store_bytes() const override { return sizeof(*this); }
    const char* name() const override { return name_; }

    double rho() const { return std::tanh(theta_); }  // copula (linear) correlation

private:
    static constexpr double kMaxTheta = 3.8002;  // atanh(0.999)

    static double sd(double var) { return std::sqrt(std::max(var, 1e-12)); }

    void update_marginals(double x, double y) {
        // Cumulative Welford (marginal scale is stationary -> no forgetting).
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
    const char* name_;
    std::size_t n_ = 0;

    double theta_ = 0.0;  // atanh(rho)
    double info_ = 0.0;   // accumulated Fisher information

    double mean_x_ = 0.0, mean_y_ = 0.0;
    double m2_x_ = 0.0, m2_y_ = 0.0;
    double var_x_ = 1.0, var_y_ = 1.0;
};
