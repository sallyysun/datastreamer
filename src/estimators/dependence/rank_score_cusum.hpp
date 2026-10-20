#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "marginal_rank_pit.hpp"

// CUSUM on the per-tick residual 12(u-1/2)(v-1/2) - rho_hat (an unbiased one-tick Spearman),
// standardized by its empirical variance; wraps any estimator via estimate().
// See notes/studies/kalman_changepoint_results.md §14.14.
class RankScoreCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    enum class Response { Hard, None };

    struct Config {
        std::size_t block = 20;
        double delta = 1.0;
        double h = 4.0;            // calibrate via calibrate_arl0.cpp
        std::size_t pit_window = 500;
        double var_rate = 0.001;   // must be far slower than the change (14.11)
        std::size_t burn_in_blocks = 5;
        Response response = Response::Hard;

        // Hold after a hard reset: inner burn-in reports rho_hat = 0, so the residual becomes rho
        // and the detector alarms forever (ARL0 = 1 block at rho = 0.8 without it).
        std::size_t resume_ticks = 200;
    };

    RankScoreCusumDetector(std::unique_ptr<DependenceEstimator> inner, Config config,
                            const char* name)
        : config_(config), name_(name), inner_(std::move(inner)),
          pit_x_(config.pit_window), pit_y_(config.pit_window) {}

    void update(double x, double y) override {
        alarmed_ = false;
        const double rho_hat = inner_->estimate();
        inner_->update(x, y);

        const double u = pit_x_.transform(x);
        const double v = pit_y_.transform(y);
        if (pit_x_.size() < 30) return;   // no usable rank scale yet
        if (hold_ > 0) { --hold_; return; }

        // rho_hat is taken BEFORE this observation is absorbed.
        const double s = 12.0 * (u - 0.5) * (v - 0.5) - rho_hat;
        var_ += config_.var_rate * (s * s - var_);

        sum_ += s / std::sqrt(std::max(var_, 1e-9));
        if (++in_block_ < config_.block) return;
        const double e = sum_ / std::sqrt(static_cast<double>(config_.block));
        sum_ = 0.0;
        in_block_ = 0;
        if (++blocks_ <= config_.burn_in_blocks) return;

        const double k = config_.delta / 2.0;
        s_pos_ = std::max(0.0, s_pos_ + e - k);
        s_neg_ = std::max(0.0, s_neg_ - e - k);

        if (s_pos_ > config_.h || s_neg_ > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            s_pos_ = s_neg_ = 0.0;
            if (config_.response == Response::Hard) {
                inner_->reset();
                hold_ = config_.resume_ticks;
                sum_ = 0.0;
                in_block_ = 0;
            }
        }
    }

    double estimate() const override { return inner_->estimate(); }

    void reset() override {
        inner_->reset();
        pit_x_.reset();
        pit_y_.reset();
        s_pos_ = s_neg_ = 0.0;
        sum_ = 0.0;
        in_block_ = 0;
        blocks_ = 0;
        var_ = 1.0;
        hold_ = 0;
        alarmed_ = false;
        alarm_count_ = 0;
    }

    std::size_t store_bytes() const override {
        return inner_->store_bytes() + pit_x_.store_bytes() + pit_y_.store_bytes()
             + sizeof(*this);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    std::size_t alarm_count() const { return alarm_count_; }

private:
    Config config_;
    const char* name_;
    std::unique_ptr<DependenceEstimator> inner_;
    RankPit pit_x_, pit_y_;

    double var_ = 1.0;
    double sum_ = 0.0;
    std::size_t in_block_ = 0;
    std::size_t blocks_ = 0;
    double s_pos_ = 0.0, s_neg_ = 0.0;
    std::size_t hold_ = 0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
};
