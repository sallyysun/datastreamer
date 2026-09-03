#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Innovation CUSUM wrapped around KalmanLocalLevel (change_detection.md Part III
// Level 1 + Part I Bridge 2). The Kalman filter already ABSORBS gradual drift --
// that is the point of Bridge 1 -- so what is left to detect is what the filter
// CANNOT absorb: a genuine level jump. This runs a classical two-sided CUSUM on
// the filter's standardized innovations e_t/sqrt(F_t), which are ~iid N(0,1)
// under the model (Basseville & Nikiforov). On alarm it calls the filter's
// reset() -- the dedicated change-detection hook already in the interface
// (estimator.hpp) -- so the next block starts from a flat prior instead of
// dragging a stale state through the jump.
//
//   S+_t = max(0, S+_{t-1} + e_t - k)     accumulates evidence of an upward jump
//   S-_t = max(0, S-_{t-1} - e_t - k)     accumulates evidence of a downward jump
//   alarm when either exceeds h
//
// k = delta/2 is the reference value for a design shift of size delta (in
// standardized-innovation units): e_t - delta/2 is exactly the log-likelihood-
// ratio increment for testing e_t ~ N(0,1) vs e_t ~ N(delta,1) (Part II of
// change_detection.md -- CUSUM = the Viterbi/max recursion over a 2-state
// absorbing chain with a known post-change mean).
//
// CAVEAT, load-bearing (see notes/todo.md): (delta, h) below are the classical
// Page / Hawkins-Olwell defaults, sanity-checked here by Monte Carlo against
// their own in-control run length on the REAL pipeline (not synthetic iid
// N(0,1) draws -- see checks/kalman_cusum_check.cpp). They are NOT calibrated
// against a target ARL0 the way change_detection.md Part VI requires before any
// detection-delay number becomes a thesis result. This class is the detector
// that calibration will be run against, not a substitute for it.
class KalmanCusumDetector : public DependenceEstimator {
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

    // sizeof(*this) already covers filter_'s fixed-size fields (it is a value
    // member, not a pointer) -- only its heap-allocated block buffers need
    // adding, same convention as KalmanLocalLevel::store_bytes() itself.
    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double);
    }

    const char* name() const override { return name_; }

    // Change-detection diagnostics -- not part of DependenceEstimator, read the
    // same way copula::CopulaFit exposes lower_tail()/upper_tail(): free once
    // computed, meant to be read once per closed block (see alarmed_this_block()).
    bool alarmed_this_block() const { return alarmed_; }
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
