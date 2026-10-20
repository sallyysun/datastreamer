#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "estimator.hpp"
#include "../objective.hpp"

// Online Gaussian-copula MLE: Fisher scoring in theta = atanh(rho), cumulative (unforgotten)
// marginal standardization. See notes/studies/roadmap.md and notes/design/source_rationale.md.
class GaussianCopulaMLE : public DependenceEstimator {
public:
    struct Config {
        double lambda = 0.0;       // RETENTION of info (larger = longer memory); 0 => cumulative
        std::size_t burn_in = 50;
        // >1: info_ /= deflation, theta_ kept (soft reset); 0: reset() is a no-op control. §14.10
        double reset_deflation = -1.0;   // <0 keeps the hard reset
        std::size_t periodic_deflate_ticks = 0;   // blind-schedule control; 0 disables
        // >0: standardize the score by its empirical variance (sandwich, §14.12). Must be much
        // slower than the change, or the scale swallows the signal (§14.11).
        double robust_score_rate = 0.0;   // 0 keeps the model information
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
        const double i_theta = 1.0 + rho * rho;
        if (!std::isfinite(s_theta)) return;

        // Standardized innovation, taken BEFORE the step.
        last_std_score_ = s_theta / std::sqrt(i_theta);
        if (config_.robust_score_rate > 0.0) {
            const double c = config_.robust_score_rate;
            score_var_ += c * (last_std_score_ * last_std_score_ - score_var_);
            last_std_score_ /= std::sqrt(std::max(score_var_, 1e-6));
        }

        const double decay = (config_.lambda > 0.0) ? config_.lambda : 1.0;
        info_ = decay * info_ + i_theta;
        // Clamp: at |rho| = 1 the step is inf and opposite-signed infinities latch theta at NaN.
        // Same +-0.999 bound as ParametricCopulaMLE.
        theta_ = std::clamp(theta_ + s_theta / info_, -kMaxTheta, kMaxTheta);

        if (config_.periodic_deflate_ticks > 0 && n_ % config_.periodic_deflate_ticks == 0) {
            info_ = std::max(info_ / config_.reset_deflation, 1e-12);
        }
    }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0;
        const double rho = std::tanh(theta_);
        return std::clamp(6.0 / std::numbers::pi * std::asin(rho / 2.0), -1.0, 1.0);
    }

    void reset() override {
        if (config_.reset_deflation == 0.0) return;
        if (config_.reset_deflation > 0.0) {
            info_ = std::max(info_ / config_.reset_deflation, 1e-12);
            return;
        }
        n_ = 0;
        theta_ = 0.0;
        info_ = 0.0;
        last_std_score_ = 0.0;
        score_var_ = 1.0;
        mean_x_ = mean_y_ = 0.0;
        m2_x_ = m2_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
    }

    std::size_t store_bytes() const override { return sizeof(*this); }
    const char* name() const override { return name_; }

    double rho() const { return std::tanh(theta_); }  // copula (linear) correlation

    // Valid only for the tick that just closed; zero during the n_ < 2 warm-up.
    double standardized_score() const { return last_std_score_; }
    std::size_t samples() const { return n_; }

private:
    static constexpr double kMaxTheta = 3.8002;  // atanh(0.999)

    static double sd(double var) { return std::sqrt(std::max(var, 1e-12)); }

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
    const char* name_;
    std::size_t n_ = 0;

    double theta_ = 0.0;
    double info_ = 0.0;
    double last_std_score_ = 0.0;
    double score_var_ = 1.0;

    double mean_x_ = 0.0, mean_y_ = 0.0;
    double m2_x_ = 0.0, m2_y_ = 0.0;
    double var_x_ = 1.0, var_y_ = 1.0;
};
