#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../estimators/dependence/estimator.hpp"
#include "../estimators/dependence/hermiter.hpp"
#include "../estimators/dependence/xiao_spearman.hpp"
#include "../protocol/market_event.hpp"
#include "ring_buffer.hpp"

// DeviceState is the recursively-updated state Z of a consumer device
// (thesis D = (Z, Psi, Z_0)). update() is the one-pass update map Psi applied
// per event. It maintains, in O(1) per update:
//   * fixed-dimensional moments  -- Welford mean/variance of prices & returns;
//   * the inference layer        -- a bank of online copula-dependence estimators.
// Folds the former running_stats / state_manager / dependence_monitor into one.
class DeviceState {
public:
    struct Moments {
        std::size_t count;
        double mean_return_x, var_return_x;
        double mean_return_y, var_return_y;
        double mean_price_x, mean_price_y;
    };

    // Estimator bank -- configs kept IDENTICAL to src/estimators/registry.hpp so
    // live (device) and offline (harness) results are directly comparable.
    DeviceState() {
        estimators_.emplace_back(std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.0, .calibration = 100}, "xiao"));
        estimators_.emplace_back(std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.99, .calibration = 100}, "xiao_ew"));
        estimators_.emplace_back(std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.0, .burn_in = 50}, "hermite"));
        estimators_.emplace_back(std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.01, .burn_in = 50}, "hermite_ew"));
    }

    // Psi: fold one event into the device state (moments + inference).
    void update(const MarketEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        return_x_.add(event.return_x);
        return_y_.add(event.return_y);
        price_x_.add(event.price_x);
        price_y_.add(event.price_y);
        for (auto& est : estimators_) est->update(event.return_x, event.return_y);
    }

    Moments moments() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Moments{
            return_x_.count(),
            return_x_.mean(), return_x_.variance(),
            return_y_.mean(), return_y_.variance(),
            price_x_.mean(), price_y_.mean(),
        };
    }

    // Inference: current Spearman's rho per estimator, in registration order.
    std::vector<std::pair<std::string, double>> dependence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, double>> out;
        out.reserve(estimators_.size());
        for (const auto& est : estimators_) out.emplace_back(est->name(), est->estimate());
        return out;
    }

    // Windowed Pearson correlation of the two return series over the most recent
    // `window` events resident in the buffer. Needs paired samples, so it reads
    // the buffer directly rather than from incremental state.
    static double rolling_correlation(const RingBuffer& buffer, std::size_t window) {
        const std::size_t head = buffer.head();
        const std::size_t oldest = buffer.oldest_available();
        std::size_t begin = (head > window) ? head - window : oldest;
        if (begin < oldest) begin = oldest;
        if (head - begin < 2) return 0.0;

        const double n = static_cast<double>(head - begin);
        double sum_x = 0.0, sum_y = 0.0;
        for (std::size_t i = begin; i < head; ++i) {
            const MarketEvent e = buffer.get(i);
            sum_x += e.return_x;
            sum_y += e.return_y;
        }
        const double mean_x = sum_x / n, mean_y = sum_y / n;
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = begin; i < head; ++i) {
            const MarketEvent e = buffer.get(i);
            const double dx = e.return_x - mean_x, dy = e.return_y - mean_y;
            cov += dx * dy; var_x += dx * dx; var_y += dy * dy;
        }
        const double denom = std::sqrt(var_x * var_y);
        return denom > 0.0 ? cov / denom : 0.0;
    }

private:
    // Online mean/variance via Welford's algorithm (was running_stats.hpp).
    struct Welford {
        void add(double x) {
            ++n_;
            const double delta = x - mean_;
            mean_ += delta / static_cast<double>(n_);
            m2_ += delta * (x - mean_);
        }
        std::size_t count() const { return n_; }
        double mean() const { return n_ > 0 ? mean_ : 0.0; }
        double variance() const { return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0; }
        std::size_t n_ = 0;
        double mean_ = 0.0, m2_ = 0.0;
    };

    mutable std::mutex mutex_;
    Welford return_x_, return_y_, price_x_, price_y_;
    std::vector<std::unique_ptr<DependenceEstimator>> estimators_;
};
