#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Upper one-sided EWMA of the squared standardized innovation (dispersion, not mean).
// See notes/studies/kalman_changepoint_results.md section 14.20.
class KalmanVarChartDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;   // must be fixed-memory: a VFF filter absorbs the over-dispersion
        double r = 0.05;
        double h = 1.5;       // calibrate via calibrate_arl0.cpp

        // > 0 replaces the EWMA with a one-sided CUSUM of (e~^2 - k).
        double cusum_k = 0.0;
    };

    KalmanVarChartDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {}

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;

        const double e = filter_.standardized_innovation();
        if (config_.cusum_k > 0.0) {
            c_ = std::max(0.0, c_ + e * e - config_.cusum_k);
        } else {
            c_ += config_.r * (e * e - c_);
        }
        ++chart_n_;

        // Baseline fixed at the model value, never re-estimated from monitored data (14.11 chasing bug);
        // its realised null level is not exactly 1, so h must come from calibration.
        if (chart_n_ > config_.kalman.burn_in_blocks && c_ > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            blocks_since_alarm_ = 0;
            filter_.reset();
            c_ = start_level();
            chart_n_ = 0;
        } else {
            ++blocks_since_alarm_;
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        c_ = start_level();
        chart_n_ = 0;
        alarmed_ = false;
        alarm_count_ = 0;
        blocks_since_alarm_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return c_; }
    std::size_t alarm_count() const { return alarm_count_; }
    std::size_t blocks_since_alarm() const { return blocks_since_alarm_; }
    double chart_statistic() const { return c_; }
    std::size_t blocks() const { return filter_.blocks(); }

private:
    double start_level() const { return config_.cusum_k > 0.0 ? 0.0 : 1.0; }

    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double c_ = start_level();
    std::size_t chart_n_ = 0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    std::size_t blocks_since_alarm_ = 0;
};
