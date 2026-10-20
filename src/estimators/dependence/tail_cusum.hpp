#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "marginal_rank_pit.hpp"
#include "../special_functions.hpp"

// Shape-change detector: tail/middle co-occurrence ratio minus its Gaussian-copula value at the
// inner rho_hat, on a Shewhart chart. See notes/studies/kalman_changepoint_results.md (§14.13, §A.8)
// and notes/design/source_rationale.md.
class TailCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        double q = 0.15;             // tail depth
        double count_rate = 0.002;   // EWMA on the two counters; ESS ~1000 ticks
        std::size_t block = 20;
        // Shewhart, not CUSUM: tail_ is already an EWMA, so a CUSUM double-counts it.
        double L = 3.0;              // calibrate via calibrate_tail.cpp
        std::size_t pit_window = 500;
        std::size_t calib_blocks = 60;   // Phase-I, then the baseline FREEZES (14.11)
    };

    TailCusumDetector(std::unique_ptr<DependenceEstimator> inner, Config config,
                       const char* name)
        : config_(config), name_(name), inner_(std::move(inner)),
          pit_x_(config.pit_window), pit_y_(config.pit_window) {}

    void update(double x, double y) override {
        alarmed_ = false;
        inner_->update(x, y);

        const double u = pit_x_.transform(x);
        const double v = pit_y_.transform(y);
        if (pit_x_.size() < config_.pit_window / 2) return;   // tail needs a real window

        const double c = config_.count_rate;
        tail_ += c * ((u <= config_.q && v <= config_.q ? 1.0 : 0.0) - tail_);
        mid_  += c * ((u <= 0.5 && v <= 0.5 ? 1.0 : 0.0) - mid_);

        const double r = tail_ / std::max(mid_, 1e-6);
        sum_ += r - gaussian_ratio(inner_->estimate());
        if (++in_block_ < config_.block) return;
        const double block_mean = sum_ / static_cast<double>(config_.block);
        sum_ = 0.0;
        in_block_ = 0;
        ++blocks_;

        if (blocks_ <= config_.calib_blocks) {
            const double n = static_cast<double>(blocks_);
            const double d = block_mean - base_mean_;
            base_mean_ += d / n;
            base_m2_ += d * (block_mean - base_mean_);
            return;
        }
        if (blocks_ == config_.calib_blocks + 1) {
            base_sd_ = std::sqrt(std::max(base_m2_ / std::max(base_mean_n() - 1.0, 1.0), 1e-12));
        }

        last_z_ = (block_mean - base_mean_) / std::max(base_sd_, 1e-9);
        if (std::fabs(last_z_) > config_.L) {
            alarmed_ = true;
            ++alarm_count_;
        }
    }

    double estimate() const override { return inner_->estimate(); }

    void reset() override {
        inner_->reset();
        pit_x_.reset();
        pit_y_.reset();
        tail_ = mid_ = 0.0;
        sum_ = 0.0;
        in_block_ = 0;
        blocks_ = 0;
        base_mean_ = base_m2_ = 0.0;
        base_sd_ = 1.0;
        last_z_ = 0.0;
        alarmed_ = false;
        alarm_count_ = 0;
    }

    std::size_t store_bytes() const override {
        return inner_->store_bytes() + pit_x_.store_bytes() + pit_y_.store_bytes()
             + sizeof(*this);
    }

    const char* name() const override { return name_; }
    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return std::fabs(last_z_); }
    std::size_t alarm_count() const { return alarm_count_; }
    double tail_ratio() const { return tail_ / std::max(mid_, 1e-6); }
    double chart_z() const { return last_z_; }
    double gaussian_ratio_for(double rho_s) const { return gaussian_ratio(rho_s); }

private:
    double base_mean_n() const { return static_cast<double>(config_.calib_blocks); }

    // Phi2(h, h; rho) via the tetrachoric identity (§A.8).
    static double bvn_diag(double h, double rho) {
        const double p = special::normal_cdf(h);
        const double r = std::clamp(rho, -0.995, 0.995);
        const int n = 200;
        const double step = r / n;
        double acc = 0.0;
        for (int i = 0; i <= n; ++i) {
            const double t = step * i;
            const double f = std::exp(-h * h / (1.0 + t))
                           / (2.0 * M_PI * std::sqrt(std::max(1.0 - t * t, 1e-12)));
            const double w = (i == 0 || i == n) ? 0.5 : 1.0;
            acc += w * f;
        }
        return p * p + acc * step;
    }

    double gaussian_ratio(double rho_s) const {
        const double rho = std::clamp(2.0 * std::sin(M_PI * rho_s / 6.0), -0.995, 0.995);
        const double zq = special::normal_ppf(config_.q);
        const double a = bvn_diag(zq, rho);
        const double b = 0.25 + std::asin(rho) / (2.0 * M_PI);
        return a / std::max(b, 1e-9);
    }

    Config config_;
    const char* name_;
    std::unique_ptr<DependenceEstimator> inner_;
    RankPit pit_x_, pit_y_;

    double tail_ = 0.0, mid_ = 0.0;
    double sum_ = 0.0;
    std::size_t in_block_ = 0, blocks_ = 0;
    double base_mean_ = 0.0, base_m2_ = 0.0, base_sd_ = 1.0;
    double last_z_ = 0.0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
};
