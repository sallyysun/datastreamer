#pragma once

#include <string>

// A single-symbol tick -- the atomic unit of the stream. One symbol, one
// instant, carrying only observables: price and volume. Returns are NOT on the
// wire; each device derives them (return = Δlog price) because price and return
// are the same information and estimation belongs on the device.
//
// Producers publish these per symbol (topic "trades.equity.<symbol>"). A device
// consumes one symbol's stream for univariate analysis (e.g. variance), or two
// for bivariate dependence (see TickPair, which is just a pair of these).
//
// Equity-only for now, and producers are assumed synchronous (one tick per
// symbol per step, same order), so arrival-order pairing is exact.
struct SymbolTick {
    std::string symbol;
    double price = 0.0;
    double volume = 0.0;
    long long timestamp = 0;

    // The DGP's true Spearman's rho for the pair this tick was generated in.
    // Only meaningful for the synthetic single-pair source (0 for real feeds);
    // carried on each leg so a consumer can validate against ground truth.
    double true_rho_s = 0.0;
    // Ground truth for scoring a synthetic run (Chapter 6): this leg's true PIT u = Phi(z), the dependence knob d of
    // this step (rho_S = (6/pi) asin(d/2)) and the copula family (gaussian|t|clayton|gumbel|gauss2t|...). Empty family =
    // no truth (real feeds).
    double true_pit = 0.0;
    double true_knob = 0.0;
    std::string true_family;
};

// Two legs at the same instant -- the bivariate observation the dependence
// estimators consume. Just a pair of SymbolTicks (there is one definition of a
// leg's fields). The DGP produces both legs correlated at the current rho; a
// consumer pairs two per-symbol SymbolTick streams back into this. Returns are
// derived on the device from each leg's price, not stored here.
struct TickPair {
    SymbolTick x;
    SymbolTick y;
};
