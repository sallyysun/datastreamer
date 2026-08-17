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
};