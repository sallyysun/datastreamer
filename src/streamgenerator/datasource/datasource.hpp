#pragma once

#include "../event/market_event.hpp"

class DataSource {
public:
    virtual ~DataSource() = default;

    virtual MarketEvent next() = 0;
};