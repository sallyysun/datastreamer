#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// EWMA control chart (Roberts 1959) on KalmanLocalLevel's standardized
// innovations; resets the filter on alarm. See
// notes/studies/kalman_changepoint_results.md.
class KalmanEwmaChartDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;
        double r = 0.2;       // chart smoothing weight
        double L = 3.0;       // control limit in SDs; calibrate via calibrate_arl0.cpp
    };

    KalmanEwmaChartDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {}

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;

        ++chart_n_;
        z_ = (1.0 - config_.r) * z_ + config_.r * filter_.standardized_innovation();

        // Time-varying limit: Var(z_n) = r/(2-r) * (1 - (1-r)^(2n)) for unit-variance
        // input. Using the asymptotic value alone makes the first few blocks far too
        // wide, which is where the BOCPD burn-in bug lived.
        const double var = config_.r / (2.0 - config_.r)
                         * (1.0 - std::pow(1.0 - config_.r, 2.0 * static_cast<double>(chart_n_)));
        const double limit = config_.L * std::sqrt(var);

        if (std::abs(z_) > limit) {
            alarmed_ = true;
            ++alarm_count_;
            blocks_since_alarm_ = 0;
            filter_.reset();
            z_ = 0.0;
            chart_n_ = 0;
        } else {
            ++blocks_since_alarm_;
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        z_ = 0.0;
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

    // |z| in units of its own time-varying SD -- the quantity actually compared
    // against L. Raw z_ is not comparable across blocks, its scale grows with
    // chart_n_.
    double monitor_statistic() const override {
        if (chart_n_ == 0) return 0.0;
        const double var = config_.r / (2.0 - config_.r)
                         * (1.0 - std::pow(1.0 - config_.r, 2.0 * static_cast<double>(chart_n_)));
        return std::abs(z_) / std::sqrt(var);
    }

    std::size_t alarm_count() const { return alarm_count_; }
    std::size_t blocks_since_alarm() const { return blocks_since_alarm_; }
    double chart_statistic() const { return z_; }
    std::size_t blocks() const { return filter_.blocks(); }

private:
    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double z_ = 0.0;
    std::size_t chart_n_ = 0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    std::size_t blocks_since_alarm_ = 0;
};
