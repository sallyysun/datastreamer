#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Two-sided CUSUM on KalmanLocalLevel's standardized innovations, which are
// ~iid N(0,1) under the model. The filter absorbs gradual drift; this detects
// what it cannot absorb. On alarm it calls the filter's reset().
//
//   S+_t = max(0, S+_{t-1} + e_t - k)
//   S-_t = max(0, S-_{t-1} - e_t - k)
//   alarm when either exceeds h
//
// k = delta/2 is the reference value for a design shift delta: e_t - delta/2 is
// the log-likelihood-ratio increment for N(0,1) vs N(delta,1). (delta, h) are
// set in registry.hpp, where h is calibrated against a target ARL0 by
// experiments/calibrate_arl0.cpp.
class KalmanCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;
        double delta = 1.0;   // design shift size, in standardized-innovation units
        double h = 5.0;       // alarm threshold
    };

    KalmanCusumDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {}

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;  // block not yet closed -- nothing new to test

        const double e = filter_.standardized_innovation();
        const double k = config_.delta / 2.0;
        s_pos_ = std::max(0.0, s_pos_ + e - k);
        s_neg_ = std::max(0.0, s_neg_ - e - k);

        if (s_pos_ > config_.h || s_neg_ > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            blocks_since_alarm_ = 0;
            filter_.reset();
            s_pos_ = s_neg_ = 0.0;
        } else {
            ++blocks_since_alarm_;
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        s_pos_ = s_neg_ = 0.0;
        alarmed_ = false;
        alarm_count_ = 0;
        blocks_since_alarm_ = 0;
    }

    // sizeof(*this) already covers filter_'s fixed-size fields; only its
    // heap-allocated block buffers need adding.
    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    std::size_t alarm_count() const { return alarm_count_; }
    std::size_t blocks_since_alarm() const { return blocks_since_alarm_; }
    double cusum_pos() const { return s_pos_; }
    double cusum_neg() const { return s_neg_; }
    std::size_t blocks() const { return filter_.blocks(); }

private:
    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double s_pos_ = 0.0, s_neg_ = 0.0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    std::size_t blocks_since_alarm_ = 0;
};
