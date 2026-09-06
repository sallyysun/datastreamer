#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "change_detector.hpp"
#include "estimator.hpp"
#include "kalman_local_level.hpp"

// Window-limited GLR on KalmanLocalLevel's standardized innovations. Where
// kalman_cusum.hpp assumes a known shift size, this handles an unknown one by
// matched-filtering the filter's own step response: a unit jump at tau
// contributes rho_s(tau) = (1-K)^{s-tau} to the innovation at s >= tau, K
// being the steady-state gain.
//
//     A(j) = sum_{i=0}^{j} (1-K)^i * e_{t-i}
//     N(j) = sum_{i=0}^{j} (1-K)^{2i}          (precomputed)
//     g_t  = max_j A(j)^2 / N(j) > h  =>  alarm; j* estimates tau,
//                                          nu_hat = A(j*)/N(j*)
//
// A(j) updates by the shift-register recurrence A_t(j) = e_t + (1-K)A_{t-1}(j-1).
// Limiting j to `window` (Lai 1995) keeps this O(window) per block; the
// unbounded sup-over-tau GLR is O(t) and not streaming-admissible.
class KalmanGlrDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        KalmanLocalLevel::Config kalman;
        std::size_t window = 30;   // max candidate lag, in blocks
        std::size_t min_lag = 1;   // shortest candidate lag (0 = a single huge innovation alone can alarm)
        double h = 10.0;           // alarm threshold on g_t
    };

    KalmanGlrDetector(Config config, const char* name)
        : config_(config), name_(name), filter_(config.kalman, name) {
        const double q = config.kalman.sigma_omega2 /
                          (1.06 / std::max(static_cast<double>(config.kalman.block) - 3.0, 1.0));
        gain_ = (std::sqrt(q * q + 4.0 * q) - q) / 2.0;  // steady-state K, closed form (Bridge 1)
        acc_.assign(config_.window, 0.0);
        weight_sq_.resize(config_.window);
        double w = 1.0, norm = 0.0;
        for (std::size_t j = 0; j < config_.window; ++j) {
            norm += w * w;
            weight_sq_[j] = norm;  // N(j) = sum_{i=0}^{j} (1-K)^{2i}
            w *= (1.0 - gain_);
        }
    }

    void update(double x, double y) override {
        const std::size_t before = filter_.blocks();
        filter_.update(x, y);
        alarmed_ = false;
        if (filter_.blocks() == before) return;  // block not yet closed

        const double e = filter_.standardized_innovation();
        // Shift-register update: A_t(j) = e_t + (1-K) A_{t-1}(j-1). Walk from
        // the top down so old_A[j-1] is read before it is overwritten.
        for (std::size_t j = config_.window - 1; j >= 1; --j) {
            acc_[j] = e + (1.0 - gain_) * acc_[j - 1];
        }
        acc_[0] = e;

        double best_g = -1.0;
        std::size_t best_j = 0;
        for (std::size_t j = config_.min_lag; j < config_.window; ++j) {
            const double g = (acc_[j] * acc_[j]) / weight_sq_[j];
            if (g > best_g) { best_g = g; best_j = j; }
        }
        last_stat_ = best_g;
        last_lag_ = best_j;
        last_nu_hat_ = acc_[best_j] / weight_sq_[best_j];

        if (best_g > config_.h) {
            alarmed_ = true;
            ++alarm_count_;
            filter_.reset();
            std::fill(acc_.begin(), acc_.end(), 0.0);
        }
    }

    double estimate() const override { return filter_.estimate(); }

    void reset() override {
        filter_.reset();
        std::fill(acc_.begin(), acc_.end(), 0.0);
        alarmed_ = false;
        alarm_count_ = 0;
        last_stat_ = 0.0;
        last_lag_ = 0;
        last_nu_hat_ = 0.0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.kalman.block * sizeof(double) + 2 * config_.window * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    std::size_t alarm_count() const { return alarm_count_; }
    double glr_statistic() const { return last_stat_; }
    std::size_t estimated_lag_blocks() const { return last_lag_; }  // blocks since the estimated tau
    double estimated_jump_size() const { return last_nu_hat_; }     // in atanh(rho) units
    double steady_state_gain() const { return gain_; }

private:
    Config config_;
    const char* name_;
    KalmanLocalLevel filter_;

    double gain_;                      // steady-state K, closed form
    std::vector<double> acc_;          // A(j), j=0..window-1
    std::vector<double> weight_sq_;    // N(j), j=0..window-1

    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
    double last_stat_ = 0.0;
    std::size_t last_lag_ = 0;
    double last_nu_hat_ = 0.0;
};
