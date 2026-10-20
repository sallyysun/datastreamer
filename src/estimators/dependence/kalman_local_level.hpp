#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "estimator.hpp"

// Local-level Kalman filter on Fisher-z'd blockwise Spearman's rho.
// See notes/studies/kalman_changepoint_results.md.
class KalmanLocalLevel : public DependenceEstimator {
public:
    struct Config {
        std::size_t block = 50;          // ticks per Spearman block; needs > 3
        double sigma_omega2 = 1e-3;      // process noise
        std::size_t burn_in_blocks = 2;
        // >0 adapts sigma_omega2 so standardized innovations have variance 1 (§0.3, §9.4).
        double vff_rate = 0.0;           // 0 disables; ~0.05 is a slow adaptation

        // >1: P_ *= inflation, z_ kept (soft reset); 0: reset() is a no-op control. §14
        double reset_inflation = -1.0;   // <0 keeps the hard reset

        // Blind-schedule control for the soft reset (§14.8).
        std::size_t periodic_inflate_blocks = 0;   // 0 disables

        double vff_min = 1e-5;
        double vff_max = 1e0;
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

        const double P_pred = P_ + sigma_omega2_;
        const double F = P_pred + sigma_v2;
        const double K = P_pred / F;
        const double innovation = y_obs - z_;

        z_ += K * innovation;
        P_ = (1.0 - K) * P_pred;

        last_innovation_ = innovation;
        last_F_ = F;
        last_K_ = K;
        ++blocks_;

        if (config_.vff_rate > 0.0) {
            // C_ tracks E[e~^2] (1 when sigma_omega2 is right); multiplicative step keeps it positive.
            const double e2 = (innovation * innovation) / std::max(F, 1e-12);
            C_ += config_.vff_rate * (e2 - C_);
            if (blocks_ > config_.burn_in_blocks) {
                sigma_omega2_ *= std::pow(std::max(C_, 1e-6), config_.vff_rate);
                sigma_omega2_ = std::clamp(sigma_omega2_, config_.vff_min, config_.vff_max);
            }
        }

        if (config_.periodic_inflate_blocks > 0 &&
            blocks_ % config_.periodic_inflate_blocks == 0) {
            P_ = std::min(P_ * config_.reset_inflation, kPriorVariance);
        }

        xs_.clear();
        ys_.clear();
    }

    double estimate() const override {
        if (blocks_ < config_.burn_in_blocks) return 0.0;
        return std::clamp(std::tanh(z_), -1.0, 1.0);
    }

    void reset() override {
        if (config_.reset_inflation == 0.0) return;
        if (config_.reset_inflation > 0.0) {
            P_ = std::min(P_ * config_.reset_inflation, kPriorVariance);
            return;
        }
        xs_.clear();
        ys_.clear();
        z_ = 0.0;
        P_ = kPriorVariance;
        blocks_ = 0;
        last_innovation_ = 0.0;
        last_F_ = kPriorVariance;
        last_K_ = 0.0;
        sigma_omega2_ = config_.sigma_omega2;
        C_ = 1.0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    double sigma_omega2() const { return sigma_omega2_; }

    // Valid only for the block that just closed; gate on blocks().
    double innovation() const { return last_innovation_; }
    double innovation_variance() const { return last_F_; }
    double standardized_innovation() const {
        return (last_F_ > 0.0) ? last_innovation_ / std::sqrt(last_F_) : 0.0;
    }
    double kalman_gain() const { return last_K_; }
    std::size_t blocks() const { return blocks_; }

private:
    static constexpr double kPriorVariance = 1.0;

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
    double sigma_omega2_ = config_.sigma_omega2;
    double C_ = 1.0;
    const char* name_;

    std::vector<double> xs_, ys_;
    double z_ = 0.0;
    double P_ = kPriorVariance;
    std::size_t blocks_ = 0;

    double last_innovation_ = 0.0;
    double last_F_ = kPriorVariance;
    double last_K_ = 0.0;
};
