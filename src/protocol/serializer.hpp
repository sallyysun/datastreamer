#pragma once

#include "symbol_tick.hpp"

#include <string>
#include <utility>
#include <vector>

// The wire is single-symbol ticks: producers publish per symbol, subscribers
// (browser or C++ consumer) pair the symbols they want.
class Serializer {
public:
    static std::string to_json(const SymbolTick& tick);
    static SymbolTick symbol_from_json(const std::string& text);

    // A consumer's dependence estimates for a pair (Phase 3: republished to the
    // broker on "estimates.equity.<PAIR>" and bridged to the browser).
    static std::string estimates_to_json(
        const std::string& pair, std::size_t n, double true_rho_s,
        const std::vector<std::pair<std::string, double>>& estimates);
};