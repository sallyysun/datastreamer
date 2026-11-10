#pragma once

#include <cstdlib>
#include <memory>
#include <string>

#include "../src/streamgenerator/datasource/dependence.hpp"

// Dependence paths for the time-varying scenarios. SCENARIOS=matched gives paths every generating family
// can attain (unrotated Clayton and Gumbel have no negative dependence): drift d = 0.2 -> 0.8 linearly in d
// over observations 1..N, oscillation d = 0.5 + 0.3 sin(2 pi t / 400). Default (original): drift -0.5 -> 0.8,
// oscillation 0.6 sin(2 pi t / 400), along which Clayton and Gumbel sit at independence wherever d < 0.
// Constant (0.6) and jump (0.2 -> 0.8) are the same in both. notes/studies/ew_marginals.md §31.
inline bool matched_scenarios() {
    static const bool v = [] { const char* e = std::getenv("SCENARIOS"); return e && std::string(e) == "matched"; }();
    return v;
}
inline std::unique_ptr<Dependence> scenario_drift(std::size_t n) {
    if (matched_scenarios()) return std::make_unique<DriftDependence>(0.2, 0.8, n - 1);   // d = 0.8 at the last tick
    return std::make_unique<DriftDependence>(-0.5, 0.8, n);
}
inline std::unique_ptr<Dependence> scenario_oscillate() {
    if (matched_scenarios()) return std::make_unique<OscillatingDependence>(0.3, 400.0, 0.5);
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}
