#pragma once

#include "symbol_tick.hpp"

#include <string>

// The wire is single-symbol ticks: producers publish per symbol, subscribers
// (browser or C++ consumer) pair the symbols they want.
class Serializer {
public:
    static std::string to_json(const SymbolTick& tick);
    static SymbolTick symbol_from_json(const std::string& text);
};