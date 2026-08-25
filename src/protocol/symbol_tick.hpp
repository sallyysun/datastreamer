#pragma once

#include <string>

// A single-symbol tick -- one leg of the market. Producers publish per symbol
// (topic "trades.equity.<symbol>"); a consumer subscribes to two symbols and
// FIFO-pairs them into a bivariate sample for its dependence estimators.
//
// Equity-only for now, and the producers are assumed to publish synchronously
// (one tick per symbol per step, same order), so arrival-order pairing is exact.
// Asynchronous / multi-rate producers are a later concern.
struct SymbolTick {
    std::string symbol;
    double price = 0.0;
    double volume = 0.0;
    double ret = 0.0;        // log return this step
    long long timestamp = 0;

    // The DGP's true Spearman's rho for the pair this tick was generated in.
    // Only meaningful for the synthetic single-pair source (0 for real feeds);
    // carried on each leg so a consumer can validate against ground truth.
    double true_rho_s = 0.0;
};
