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
#include "../protocol/symbol_tick.hpp"

// Device state: Welford moments of prices/returns + the online dependence-estimator bank, O(1)/tick.
// Thesis framing (D = (Z, Psi, Z_0)): notes/design/source_rationale.md.
class DeviceState {
public:
    struct Moments {
        std::size_t count;
        double mean_return_x, var_return_x;
        double mean_return_y, var_return_y;
        double mean_price_x, mean_price_y;
    };

    // Configs must stay IDENTICAL to src/estimators/registry.hpp (live vs offline RMSE comparability).
    // λ is a mirror-image knob: Xiao larger = longer memory, Hermite smaller = longer; see registry.hpp.
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

    // Wire carries prices; return = ln(p_t / p_{t-1}), so the first event only seeds prev price.
    void update(const TickPair& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        price_x_.add(event.x.price);
        price_y_.add(event.y.price);
        if (has_prev_) {
            const double rx = std::log(event.x.price / prev_price_x_);
            const double ry = std::log(event.y.price / prev_price_y_);
            return_x_.add(rx);
            return_y_.add(ry);
            for (auto& est : estimators_) est->update(rx, ry);
        }
        prev_price_x_ = event.x.price;
        prev_price_y_ = event.y.price;
        has_prev_ = true;
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

    std::vector<std::pair<std::string, double>> dependence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, double>> out;
        out.reserve(estimators_.size());
        for (const auto& est : estimators_) out.emplace_back(est->name(), est->estimate());
        return out;
    }

private:
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
    double prev_price_x_ = 0.0, prev_price_y_ = 0.0;
    bool has_prev_ = false;
    std::vector<std::unique_ptr<DependenceEstimator>> estimators_;
};
