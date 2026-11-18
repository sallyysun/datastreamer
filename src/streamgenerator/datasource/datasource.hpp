#pragma once

#include "../../protocol.hpp"

class DataSource {
public:
    virtual ~DataSource() = default;

    virtual TickPair next() = 0;
};