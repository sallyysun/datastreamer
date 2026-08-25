#pragma once

#include "../../protocol/symbol_tick.hpp"

class DataSource {
public:
    virtual ~DataSource() = default;

    virtual TickPair next() = 0;
};