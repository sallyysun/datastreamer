#pragma once

// The thesis estimators (Chapter 4) and forgetting strategies (Chapter 5), configured once.
//   estimators: "rmpl" (recursive Gaussian MPL), "xiao" (Counting Matrix), "hermite" (Hermite Series)
//   strategies: fixed memory N_eff (arm), VFF (prediction-error controller), MFF (Hedge + fixed share, log loss)
//
// ⚠️ The forgetting knob is a mirror image across estimators, all mapped from one memory N_eff:
//   Counting Matrix retention lambda = (N - 1)/(N + 1)   (LARGER = longer memory)
//   Hermite update weight     lambda = 2/(N + 1)         (SMALLER = longer memory)
//   RMPL gain                        = 2/(N + 1)
// N_eff = 0 means no forgetting (cumulative).
//
// The live consumer (src/consumer/) builds its estimators from these functions, so offline and live
// numbers share one configuration.

#include <memory>
#include <string>
#include <vector>

#include "dependence/estimator.hpp"
#include "dependence/forgetting_bank.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/recursive_mpl.hpp"
#include "dependence/vff.hpp"
#include "dependence/xiao_spearman.hpp"

namespace thesis {

inline const std::vector<std::string>& estimators() {
    static const std::vector<std::string> v = {"rmpl", "xiao", "hermite"};
    return v;
}
inline const std::vector<double>& mff_memories() {
    static const std::vector<double> v = {39, 99, 199, 399, 1999};
    return v;
}
constexpr double kVffRate = 0.02;
constexpr double kMffEta = 1.0, kMffDelta = 1e-4;

// shared: train on the shared normal scores probit(u) rather than on raw observations (needed to score an
// arm's own density at the shared pseudo-observations).
inline XiaoSpearmanEstimator::Config xiao(double lambda, bool shared = false) {
    return XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = lambda, .calibration = 100,
                                         .normal_score_inputs = shared};
}
inline HermiteEstimator::Config hermite(double lambda, bool shared = false) {
    return HermiteEstimator::Config{.order = 10, .lambda = lambda, .burn_in = 0, .normal_score_inputs = shared};
}
// Input is z = probit(u) of the shared pseudo-observations; the start-up gain is max(gain, 1/m).
inline RecursiveMPL::Config rmpl(double gain) {
    RecursiveMPL::Config c{};
    c.margin = RecursiveMPL::Margin::ExactNormal;
    c.gain = gain;
    c.startup = true;
    c.startup_normalised = true;
    return c;
}

// One estimator at fixed memory N_eff (0 = no forgetting).
inline std::unique_ptr<DependenceEstimator> arm(const std::string& est, double neff, const char* name = "arm") {
    const bool nf = neff == 0;
    if (est == "xiao") return std::make_unique<XiaoSpearmanEstimator>(xiao(nf ? 0.0 : (neff - 1) / (neff + 1)), name);
    if (est == "hermite") return std::make_unique<HermiteEstimator>(hermite(nf ? 0.0 : 2 / (neff + 1)), name);
    if (est == "rmpl") return std::make_unique<RecursiveMPL>(rmpl(nf ? 1e-12 : 2 / (neff + 1)), name);
    return nullptr;
}

// Variable forgetting factor: one estimator whose N_eff follows the prediction-error controller.
inline std::unique_ptr<DependenceEstimator> vff(const std::string& est, double rate = kVffRate, const char* name = "vff") {
    auto inner = arm(est, 199, "inner");
    if (!inner) return nullptr;
    return std::make_unique<CommonVff>(std::move(inner), VffController::Config{.rate = rate}, est != "rmpl", name);
}

// Multiple forgetting factors: Hedge with fixed share over the five fixed-memory arms, log-loss weights,
// mixture-density forecast.
inline std::unique_ptr<DependenceEstimator> mff(const std::string& est, const char* name = "mff") {
    std::vector<std::unique_ptr<DependenceEstimator>> arms;
    for (double n : mff_memories()) arms.push_back(arm(est, n, "m"));
    if (!arms[0]) return nullptr;
    ForgettingBank::Config c{.eta = kMffEta, .delta = kMffDelta};
    c.loss = ForgettingBank::Loss::Log;
    c.arms_raw = est != "rmpl";
    return std::make_unique<ForgettingBank>(std::move(arms), c, name);
}

}  // namespace thesis
