#pragma once

#include <string>

// One symbol's price and volume at a given time, with ground truth for synthetic streams.
// Consumers pair synchronous ticks and derive log returns from prices.
struct SymbolTick {
    std::string symbol;
    double price = 0.0;
    double volume = 0.0;
    long long timestamp = 0;
    double true_rho_s = 0.0;
    double true_pit = 0.0;
    double true_knob = 0.0;
    std::string true_family;
};

struct TickPair {
    SymbolTick x;
    SymbolTick y;
};
