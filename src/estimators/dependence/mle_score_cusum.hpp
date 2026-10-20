#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "gaussian_mle.hpp"

// Two-sided CUSUM on the Gaussian-copula MLE's standardized score s / sqrt(I_theta), block-summed.
// See section 14.10 of notes/studies/kalman_changepoint_results.md.
class MleScoreCusumDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        GaussianCopulaMLE::Config mle;
        std::size_t block = 20;
        double delta = 1.0;   // in block-score SDs
        double h = 3.5;       // calibrate via calibrate_arl0.cpp
        std::size_t burn_in_blocks = 3;
    };

    MleScoreCusumDetector(Config config, const char* name)
        : config_(config), name_(name), inner_(config.mle, name) {}

    void update(double x, double y) override {
        inner_.update(x, y);
        alarmed_ = false;

        sum_ += inner_.standardized_score();
        if (++in_block_ < config_.block) return;

        // Unit-variance, weakly dependent terms: scale the block sum by sqrt(block).
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
            inner_.reset();
            s_pos_ = s_neg_ = 0.0;
        }
    }

    double estimate() const override { return inner_.estimate(); }

    void reset() override {
        inner_.reset();
        s_pos_ = s_neg_ = 0.0;
        sum_ = 0.0;
        in_block_ = 0;
        blocks_ = 0;
        alarmed_ = false;
        alarm_count_ = 0;
    }

    std::size_t store_bytes() const override {
        return inner_.store_bytes() + sizeof(*this) - sizeof(inner_);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    double monitor_statistic() const override { return std::max(s_pos_, s_neg_); }
    std::size_t alarm_count() const { return alarm_count_; }

private:
    Config config_;
    const char* name_;
    GaussianCopulaMLE inner_;

    double sum_ = 0.0;
    std::size_t in_block_ = 0;
    std::size_t blocks_ = 0;
    double s_pos_ = 0.0, s_neg_ = 0.0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
};
