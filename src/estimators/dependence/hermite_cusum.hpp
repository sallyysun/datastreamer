#pragma once

#include <algorithm>
#include <cmath>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "hermiter.hpp"

// One-sided CUSUM on HermiteEstimator's coefficient-matrix surprise, rather than
// on a scalar rho. Detects a change of copula *shape* at constant rho, which no
// rho-monitoring detector can see. See notes/studies/kalman_changepoint_results.md
// section 0.4 (distinction 1).
class HermiteCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        HermiteEstimator::Config hermite;
        double delta = 1.0;   // design shift, in surprise SDs
        double h = 5.0;       // alarm threshold; calibrate via calibrate_arl0.cpp
        // Surprises are averaged over a block before being thresholded. Per tick
        // the statistic is a rank-1 outer-product norm, dominated by the single
        // observation's magnitude; a block mean is what has usable SNR, and
        // block = 20 makes the alarm rate comparable to every other detector's.
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

        // One-sided: only a *rise* in surprise indicates the coefficients have
        // moved away from what the estimator has absorbed.
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
