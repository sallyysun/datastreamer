#pragma once

// The thesis estimators (Chapter 4), configured once: "rmpl" (recursive Gaussian MPL), "xiao" (Counting Matrix),
// "hermite" (Hermite Series). The forgetting strategies (Chapter 5: fixed N_eff, VFF, MFF) and the Chapter 6
// device built from them are in evaluation.hpp and device.hpp.
//
// ⚠️ The forgetting knob is a mirror image across estimators, all mapped from one memory N_eff:
//   Counting Matrix retention lambda = (N - 1)/(N + 1)   (LARGER = longer memory)
//   Hermite update weight     lambda = 2/(N + 1)         (SMALLER = longer memory)
//   RMPL gain                        = 2/(N + 1)
// N_eff = 0 means no forgetting (cumulative).

#include <memory>
#include <string>
#include <vector>

#include "dependence/estimator.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/rmpl.hpp"
#include "dependence/count_matrix.hpp"

namespace thesis {

inline const std::vector<std::string>& estimators() {
    static const std::vector<std::string> v = {"xiao", "hermite", "rmpl"};
    return v;
}

// shared: train on the shared normal scores probit(u) rather than on the raw observations.
inline CountingMatrixEstimator::Config counting_matrix_config(double lambda, bool shared = false) {
    return CountingMatrixEstimator::Config{.bins = 20, .lambda = lambda, .calibration = 100, .normal_score_inputs = shared};
}
inline HermiteEstimator::Config hermite(double lambda, bool shared = false) {
    return HermiteEstimator::Config{.order = 10, .lambda = lambda, .normal_score_inputs = shared};
}
// Input is z = probit(u) of the shared pseudo-observations.
inline RecursiveMPL::Config rmpl(double gain) {
    RecursiveMPL::Config c{};
    c.margin = RecursiveMPL::Margin::ExactNormal;
    c.gain = gain;
    return c;
}

// One estimator at fixed memory N_eff (0 = no forgetting).
inline std::unique_ptr<DependenceEstimator> arm(const std::string& est, double neff, const char* name = "arm") {
    const bool nf = neff == 0;
    if (est == "xiao") return std::make_unique<CountingMatrixEstimator>(counting_matrix_config(nf ? 0.0 : (neff - 1) / (neff + 1)), name);
    if (est == "hermite") return std::make_unique<HermiteEstimator>(hermite(nf ? 0.0 : 2 / (neff + 1)), name);
    if (est == "rmpl") return std::make_unique<RecursiveMPL>(rmpl(nf ? 1e-12 : 2 / (neff + 1)), name);
    return nullptr;
}

}  // namespace thesis
