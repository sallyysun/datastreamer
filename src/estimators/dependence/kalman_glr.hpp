#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Window-limited GLR on KalmanLocalLevel's innovations for an unknown jump size; resets on alarm.
// See notes/studies/kalman_changepoint_results.md.
class KalmanGlrDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;
        std::size_t window = 30;   // max candidate jump lag, in blocks
        std::size_t min_lag = 1;
        double h = 10.0;           // calibrate via calibrate_arl0.cpp
    };

    KalmanGlrDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {
        const double q = config.kalman.sigma_omega2 /
                          (1.06 / std::max(static_cast<double>(config.kalman.block) - 3.0, 1.0));
        gain_ = (std::sqrt(q * q + 4.0 * q) - q) / 2.0;
        acc_.assign(config_.window, 0.0);
        weight_sq_.resize(config_.window);
        double w = 1.0, norm = 0.0;
        for (std::size_t j = 0; j < config_.window; ++j) {
            norm += w * w;
            weight_sq_[j] = norm;
            w *= (1.0 - gain_);
        }
    }

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;

        const double e = filter_.standardized_innovation();
        // Walk down so acc_[j-1] is read before being overwritten.
        for (std::size_t j = config_.window - 1; j >= 1; --j) {
            acc_[j] = e + (1.0 - gain_) * acc_[j - 1];
        }
        acc_[0] = e;

        double best_g = -1.0;
        std::size_t best_j = 0;
        for (std::size_t j = config_.min_lag; j < config_.window; ++j) {
            const double g = (acc_[j] * acc_[j]) / weight_sq_[j];
            if (g > best_g) { best_g = g; best_j = j; }
        }
        last_stat_ = best_g;
        last_lag_ = best_j;
        last_nu_hat_ = acc_[best_j] / weight_sq_[best_j];

        if (best_g > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            filter_.reset();
            std::fill(acc_.begin(), acc_.end(), 0.0);
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        std::fill(acc_.begin(), acc_.end(), 0.0);
        alarmed_ = false;
        alarm_count_ = 0;
        last_stat_ = 0.0;
        last_lag_ = 0;
        last_nu_hat_ = 0.0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double) + 2 * config_.window * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return last_stat_; }
    std::size_t alarm_count() const { return alarm_count_; }
    double glr_statistic() const { return last_stat_; }
    std::size_t estimated_lag_blocks() const { return last_lag_; }
    double estimated_jump_size() const { return last_nu_hat_; }  // atanh units
    double steady_state_gain() const { return gain_; }

private:
    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double gain_;
    std::vector<double> acc_;
    std::vector<double> weight_sq_;

    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    double last_stat_ = 0.0;
    std::size_t last_lag_ = 0;
    double last_nu_hat_ = 0.0;
};
