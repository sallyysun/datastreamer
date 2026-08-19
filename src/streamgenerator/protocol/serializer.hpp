#pragma once

#include "../event/market_event.hpp"

#include <string>
#include <utility>
#include <vector>

class Serializer {
public:
    static std::string to_json(const MarketEvent& event);

    // Same payload plus a "dependence" object mapping estimator name -> its
    // current Spearman rho estimate, so live estimates reach the client.
    static std::string to_json(
        const MarketEvent& event,
        const std::vector<std::pair<std::string, double>>& dependence);

    // Inverse of to_json: parse one JSON tick back into a MarketEvent. Used by
    // C++ consumers reading the raw event stream off the broker.
    static MarketEvent from_json(const std::string& text);
};