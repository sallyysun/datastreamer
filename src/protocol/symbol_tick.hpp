#pragma once

#include <string>

// Wire tick (topic "trades.equity.<symbol>"): observables only. Returns are not on the wire;
// devices derive them as Δlog price. Producers are assumed synchronous (one tick per symbol per
// step, same order), so arrival-order pairing of two symbols into a TickPair is exact.
struct SymbolTick {
    std::string symbol;
    double price = 0.0;
    double volume = 0.0;
    long long timestamp = 0;

    // DGP ground-truth Spearman's rho for the pair (0 for real feeds).
    double true_rho_s = 0.0;
};

struct TickPair {
    SymbolTick x;
    SymbolTick y;
};
