#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "estimator.hpp"

// Local-level Kalman filter on Fisher-z'd blockwise Spearman's rho.
// See notes/studies/change_detection.md, Part I "Bridge 1" and Part III Level 1.
//
// State-space model on z_t = atanh(rho_t):
//   z_t = z_{t-1} + omega_t,   omega_t ~ N(0, sigma_omega^2)   (state: dependence wanders)
//   y_t = z_t + v_t,           v_t     ~ N(0, sigma_v^2)       (observation: a noisy read)
//
// This is a STRICT GENERALIZATION of every fixed-lambda estimator in this
// directory, not a competing branch: at steady state the Kalman gain K_t -> a
// constant K, and the recursion collapses to an EWMA z_t = (1-K) z_{t-1} + K y_t,
// i.e. lambda = 1-K. Choosing sigma_omega^2 (equivalently the SNR
// q = sigma_omega^2/sigma_v^2) replaces "pick a lambda by hand" with "pick a
// process-noise variance with a physical meaning" -- see change_detection.md for
// the closed-form K(q) used to cross-check this file's steady-state behaviour.
//
// Why blockwise, not per-tick: the per-tick normal-score product z_x*z_y has
// mean rho and variance 1+rho^2, and the link rho=tanh(z) is nonlinear, so an
// exact one-tick Kalman filter would need an EKF/UKF. Instead this emits
// y_t = atanh(windowed Spearman rho) every `block` ticks. The sampling variance
// of a Fisher-z'd Spearman correlation over m points is KNOWN in closed form,
// sigma_v^2 ~= 1.06/(m-3) (the classical Fieller-Hartley-Pearson approximation,
// inflated 6% relative to Pearson's 1/(n-3) for the same reason Spearman's own
// asymptotic variance is inflated over Pearson's) -- so within a block this
// stays an EXACT linear-Gaussian filter, no EKF needed, and no free variance
// parameter to fit.
//
// The single free knob is sigma_omega^2. Between block boundaries there is no
// new observation, so estimate() simply holds the last filtered z_t flat
// (E[omega]=0 -- the predicted mean does not move even though the predicted
// variance P_{t|t-1} would, if we tracked it past the next block).
class KalmanLocalLevel : public DependenceEstimator {
public:
    struct Config {
        std::size_t block = 50;         // ticks per Spearman block (m); needs m>3
        double sigma_omega2 = 1e-3;     // process noise -- the free knob (the "q")
        std::size_t burn_in_blocks = 2;  // blocks before reporting a non-zero estimate
    };

    KalmanLocalLevel(Config config, const char* name)
        : config_(config), name_(name) {
        xs_.reserve(config_.block);
        ys_.reserve(config_.block);
    }

    void update(double x, double y) override {
        xs_.push_back(x);
        ys_.push_back(y);
        if (xs_.size() < config_.block) return;

        const double rho_s = block_spearman();
        const double y_obs = std::atanh(std::clamp(rho_s, -0.999, 0.999));
        const double m = static_cast<double>(config_.block);
        const double sigma_v2 = 1.06 / std::max(m - 3.0, 1.0);

        // Predict: one state-transition step since the last observation.
        const double P_pred = P_ + config_.sigma_omega2;
        // Update.
        const double F = P_pred + sigma_v2;
        const double K = P_pred / F;
        const double innovation = y_obs - z_;

        z_ += K * innovation;
        P_ = (1.0 - K) * P_pred;

        last_innovation_ = innovation;
        last_F_ = F;
        last_K_ = K;
        ++blocks_;

        xs_.clear();
        ys_.clear();
    }

    double estimate() const override {
        if (blocks_ < config_.burn_in_blocks) return 0.0;
        return std::clamp(std::tanh(z_), -1.0, 1.0);
    }

    void reset() override {
        xs_.clear();
        ys_.clear();
        z_ = 0.0;
        P_ = kPriorVariance;
        blocks_ = 0;
        last_innovation_ = 0.0;
        last_F_ = kPriorVariance;
        last_K_ = 0.0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    // Change-detection hooks -- not part of DependenceEstimator, exposed the same
    // way ParametricCopulaMLE exposes lower_tail()/upper_tail(): free once fitted,
    // and the exact quantities an innovation-CUSUM (change_detection.md Part III
    // Level 1) is built on. Only meaningful right after a block just closed;
    // NaN between block boundaries would be more honest but adds a branch every
    // caller needs to handle, so callers should read these once per `block` ticks
    // (e.g. gate on blocks_changed()).
    double innovation() const { return last_innovation_; }
    double innovation_variance() const { return last_F_; }
    double standardized_innovation() const {
        return (last_F_ > 0.0) ? last_innovation_ / std::sqrt(last_F_) : 0.0;
    }
    double kalman_gain() const { return last_K_; }
    std::size_t blocks() const { return blocks_; }

private:
    static constexpr double kPriorVariance = 1.0;  // weak prior on z_0

    double block_spearman() const {
        const std::size_t n = xs_.size();
        const std::vector<double> rx = average_ranks(xs_);
        const std::vector<double> ry = average_ranks(ys_);

        const double mean = static_cast<double>(n + 1) / 2.0;
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = rx[i] - mean;
            const double dy = ry[i] - mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        const double denom = std::sqrt(var_x * var_y);
        return (denom > 0.0) ? cov / denom : 0.0;
    }

    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&v](std::size_t a, std::size_t b) -> bool { return v[a] < v[b]; });

        std::vector<double> ranks(n);
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            while (j + 1 < n && v[order[j + 1]] == v[order[i]]) ++j;
            const double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
            for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = avg;
            i = j + 1;
        }
        return ranks;
    }

    Config config_;
    const char* name_;

    std::vector<double> xs_, ys_;
    double z_ = 0.0;              // filtered state, atanh(rho)
    double P_ = kPriorVariance;   // filtered state variance
    std::size_t blocks_ = 0;

    double last_innovation_ = 0.0;
    double last_F_ = kPriorVariance;
    double last_K_ = 0.0;
};
