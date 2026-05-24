#pragma once

#include "../event/market_event.hpp"

#include <string>

class Serializer {
public:
    static std::string to_json(const MarketEvent& event);
};