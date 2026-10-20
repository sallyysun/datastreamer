#pragma once

#include "symbol_tick.hpp"

#include <string>

// JSON wire format: one single-symbol tick per message; subscribers pair the legs themselves.
class Serializer {
public:
    static std::string to_json(const SymbolTick& tick);
    static SymbolTick symbol_from_json(const std::string& text);
};