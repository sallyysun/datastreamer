#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>

#include "../estimators/thesis_device.hpp"
#include "../protocol.hpp"
#include "paired_returns.hpp"

// Track price and return moments alongside the Chapter 6 estimators.
class DeviceState {
public:
    struct Moments {
        std::size_t count;
        double mean_return_x, var_return_x;
        double mean_return_y, var_return_y;
        double mean_price_x, mean_price_y;
    };

    DeviceState() = default;

    // Score every strategy against the ground truth carried on the ticks (rows as the runner's strategies_seeds).
    void enable_scoring(const std::string& scenario) { scenario_ = scenario; }

    // Fold prices, returns and forecasts into the device state.
    void update(const TickPair& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        price_x_.add(event.x.price);
        price_y_.add(event.y.price);
        returns_.on_tick(event, [&](double rx, double ry) {
            return_x_.add(rx);
            return_y_.add(ry);
            if (scorer_ || (!scenario_.empty() && thesis::scorable(event.x.true_family))) {
                if (!scorer_) scorer_ = std::make_unique<thesis::Scorer>(event.x.true_family, scenario_);
                scorer_->observe(dev_, dev_.count(), event.x.true_rho_s, event.x.true_pit, event.y.true_pit,
                                 thesis::true_param(event.x.true_family, event.x.true_target));
            }
            dev_.update(rx, ry);
        });
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

    // Inference: current rho_S forecast per estimator x strategy ("rmpl_mff", "xiao_fixed_39", ...).
    std::vector<std::pair<std::string, double>> dependence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, double>> out;
        std::vector<double> f;
        for (const auto& set : dev_.sets()) {
            set.forecasts(f);
            for (std::size_t i = 0; i < f.size(); ++i) out.emplace_back(set.name() + "_" + thesis::strategy_names()[i], f[i]);
        }
        return out;
    }

    std::size_t observations() const { std::lock_guard<std::mutex> lock(mutex_); return dev_.count(); }
    void write_scores(std::FILE* out, unsigned seed) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (scorer_) scorer_->write(out, dev_, seed);
    }

private:
    // Welford running mean and variance.
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
    PairedReturns returns_;
    thesis::Device dev_;
    std::string scenario_;
    std::unique_ptr<thesis::Scorer> scorer_;
};
