#pragma once

#include <algorithm>
#include <cmath>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "hermiter.hpp"

// One-sided CUSUM on HermiteEstimator's coefficient-matrix surprise: detects a copula shape change
// at constant rho. See notes/studies/kalman_changepoint_results.md section 0.4 (distinction 1).
class HermiteCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        HermiteEstimator::Config hermite;
        double delta = 1.0;   // in surprise SDs
        double h = 5.0;       // calibrate via calibrate_arl0.cpp
        // Per-tick surprise is a rank-1 norm with no usable SNR; only the block mean is thresholded.
        std::size_t block = 20;
    };

    HermiteCusumDetector(Config config, const char* name)
        : config_(config), name_(name), inner_(config.hermite, name) {}

    void update(double x, double y) override {
        inner_.update(x, y);
        alarmed_ = false;

        const double e_tick = inner_.standardized_surprise();
        if (e_tick == 0.0) return;   // still inside the surprise burn-in

        sum_ += e_tick;
        if (++in_block_ < config_.block) return;
        const double e = sum_ / std::sqrt(static_cast<double>(config_.block));
        sum_ = 0.0;
        in_block_ = 0;

        s_ = std::max(0.0, s_ + e - config_.delta / 2.0);

        if (s_ > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            s_ = 0.0;
            inner_.reset();
        }
    }

    double estimate() const override { return inner_.estimate(); }

    void reset() override {
        inner_.reset();
        s_ = 0.0;
        sum_ = 0.0;
        in_block_ = 0;
        alarmed_ = false;
        alarm_count_ = 0;
    }

    std::size_t store_bytes() const override {
        return inner_.store_bytes() + sizeof(*this) - sizeof(inner_);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return s_; }
    std::size_t alarm_count() const { return alarm_count_; }

private:
    Config config_;
    const char* name_;
    HermiteEstimator inner_;
    double s_ = 0.0;
    double sum_ = 0.0;
    std::size_t in_block_ = 0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
};
