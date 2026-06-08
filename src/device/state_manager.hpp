#pragma once

#include <cmath>
#include <cstddef>
#include <mutex>

#include "../streamgenerator/event/market_event.hpp"
#include "ring_buffer.hpp"
#include "running_stats.hpp"

// Holds the current statistical state for one consumer.
//
// Two flavours of estimator live here:
//
//  1. Incremental ones (mean, variance) are folded in on every event via
//     update() and kept in O(1) RunningStats -- this is "the current state".
//
//  2. Windowed estimators that need more than one value at once (e.g. a
//     rolling correlation) can't be maintained from a single scalar, so they
//     read straight from the ring buffer over the most recent window.
class StateManager {
public:
    struct Snapshot {
        std::size_t count;
        double mean_return_x;
        double var_return_x;
        double mean_return_y;
        double var_return_y;
        double mean_price_x;
        double mean_price_y;
    };

    void update(const MarketEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        return_x_.add(event.return_x);
        return_y_.add(event.return_y);
        price_x_.add(event.price_x);
        price_y_.add(event.price_y);
    }

    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot{
            return_x_.count(),
            return_x_.mean(), return_x_.variance(),
            return_y_.mean(), return_y_.variance(),
            price_x_.mean(), price_y_.mean(),
        };
    }

    // Pearson correlation of the two return series over the most recent
    // `window` events still resident in the buffer. Needs paired samples, so
    // it reads the buffer directly rather than from incremental state.
    static double rolling_correlation(const RingBuffer& buffer, std::size_t window) {
        const std::size_t head = buffer.head();
        const std::size_t oldest = buffer.oldest_available();

        std::size_t begin = (head > window) ? head - window : oldest;
        if (begin < oldest) {
            begin = oldest;
        }
        if (head - begin < 2) {
            return 0.0;
        }

        const double n = static_cast<double>(head - begin);

        double sum_x = 0.0, sum_y = 0.0;
        for (std::size_t i = begin; i < head; ++i) {
            const MarketEvent e = buffer.get(i);
            sum_x += e.return_x;
            sum_y += e.return_y;
        }
        const double mean_x = sum_x / n;
        const double mean_y = sum_y / n;

        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = begin; i < head; ++i) {
            const MarketEvent e = buffer.get(i);
            const double dx = e.return_x - mean_x;
            const double dy = e.return_y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        const double denom = std::sqrt(var_x * var_y);
        return denom > 0.0 ? cov / denom : 0.0;
    }

private:
    mutable std::mutex mutex_;
    RunningStats return_x_;
    RunningStats return_y_;
    RunningStats price_x_;
    RunningStats price_y_;
};
