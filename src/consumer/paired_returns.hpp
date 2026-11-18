#pragma once

#include <cmath>
#include <optional>
#include <utility>

#include "../protocol.hpp"

class PairedReturns {
public:
    // The opening tick supplies prices for the next return.
    template <typename Observe>
    void on_tick(const TickPair& event, Observe observe) {
        const std::pair prices{event.x.price, event.y.price};
        if (previous_) {
            const double x = std::log(prices.first / previous_->first);
            const double y = std::log(prices.second / previous_->second);
            observe(x, y);
        }
        previous_ = prices;
    }

private:
    std::optional<std::pair<double, double>> previous_;
};
