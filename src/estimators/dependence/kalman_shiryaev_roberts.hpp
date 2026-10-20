#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Shiryaev-Roberts, R_t = (1 + R_{t-1}) * Lambda_t carried in logs, on KalmanLocalLevel's
// standardized innovations; resets the filter on alarm. See notes/studies/kalman_changepoint_results.md.
class KalmanShiryaevRobertsDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;
        double delta = 1.0;   // in innovation SDs
        double h = 5.0;       // on log R; calibrate via calibrate_arl0.cpp
    };

    KalmanShiryaevRobertsDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {}

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;

        const double e = filter_.standardized_innovation();
        const double d = config_.delta;

        log_r_pos_ = accumulate(log_r_pos_, d * e - 0.5 * d * d);
        log_r_neg_ = accumulate(log_r_neg_, -d * e - 0.5 * d * d);

        if (log_r_pos_ > config_.h || log_r_neg_ > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            blocks_since_alarm_ = 0;
            filter_.reset();
            log_r_pos_ = log_r_neg_ = kLogZero;
        } else {
            ++blocks_since_alarm_;
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        log_r_pos_ = log_r_neg_ = kLogZero;
        alarmed_ = false;
        alarm_count_ = 0;
        blocks_since_alarm_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return log_statistic(); }
    std::size_t alarm_count() const { return alarm_count_; }
    std::size_t blocks_since_alarm() const { return blocks_since_alarm_; }
    double log_statistic() const { return std::max(log_r_pos_, log_r_neg_); }
    std::size_t blocks() const { return filter_.blocks(); }

private:
    static constexpr double kLogZero = -1e300;

    // log1p on the exponentiated previous value so R = 0 is representable without underflow.
    static double accumulate(double log_r_prev, double log_lambda) {
        const double log1p_r = (log_r_prev <= kLogZero / 2.0)
                             ? 0.0
                             : ((log_r_prev > 30.0) ? log_r_prev
                                                    : std::log1p(std::exp(log_r_prev)));
        return log1p_r + log_lambda;
    }

    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double log_r_pos_ = kLogZero, log_r_neg_ = kLogZero;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    std::size_t blocks_since_alarm_ = 0;
};
