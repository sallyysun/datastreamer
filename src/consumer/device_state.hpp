#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>

#include "../estimators/thesis_device.hpp"
#include "../protocol/symbol_tick.hpp"

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

    // Inference = the Chapter 6 device (src/estimators/thesis_device.hpp): shared KLL pseudo-observations, and per
    // estimator the five fixed arms, MFF and VFF, exactly as the in-process probe.
    DeviceState() = default;

    // Score every strategy against the ground truth carried on the ticks (rows as the probe's strategies_seeds).
    void enable_scoring(const std::string& scenario) { scenario_ = scenario; }

    // Psi: fold one event into the device state (moments + inference). The wire
    // carries prices; the return is derived here as the log-price increment
    // (return = ln(price_t / price_{t-1})), so the first event only seeds the
    // previous price and produces no return.
    void update(const TickPair& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        price_x_.add(event.x.price);
        price_y_.add(event.y.price);
        if (has_prev_) {
            const double rx = std::log(event.x.price / prev_price_x_);
            const double ry = std::log(event.y.price / prev_price_y_);
            return_x_.add(rx);
            return_y_.add(ry);
            if (!scenario_.empty() && !event.x.true_family.empty()) {
                if (!scorer_) scorer_ = std::make_unique<thesis::Scorer>(event.x.true_family, scenario_);
                scorer_->observe(dev_, dev_.count(), event.x.true_rho_s, event.x.true_pit, event.y.true_pit,
                                 event.x.true_param);
            }
            dev_.update(rx, ry);
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
    double prev_price_x_ = 0.0, prev_price_y_ = 0.0;  // for return = Δlog price
    bool has_prev_ = false;
    thesis::Device dev_;
    std::string scenario_;
    std::unique_ptr<thesis::Scorer> scorer_;
};
