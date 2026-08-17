#pragma once

#include <memory>
#include <string>

#include "dependence/estimator.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/windowed_rank.hpp"
#include "dependence/xiao_spearman.hpp"

// Named constructors for the nonparametric dependence estimators so experiments
// can be configured by string instead of recompiled. Returns nullptr for an
// unknown key.
//
// Forgetting conventions differ per estimator (documented here so configs are
// comparable):
//   * Xiao   lambda = retention factor in (0,1); larger = longer memory.
//            Effective sample size ~ (1+lambda)/(1-lambda).
//   * Hermite lambda = EWMA smoothing in (0,1); larger = faster adaptation.
//            Effective sample size ~ 1/lambda. lambda = 0 => cumulative.
inline std::unique_ptr<DependenceEstimator> make_estimator(const std::string& key) {
    if (key == "xiao") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{}, "xiao");
    }
    if (key == "xiao_ew") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 30, .window = 0, .lambda = 0.99, .calibration = 300},
            "xiao_ew");
    }
    if (key == "xiao_win") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 30, .window = 500, .lambda = 0.0, .calibration = 300},
            "xiao_win");
    }
    if (key == "hermite") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{}, "hermite");
    }
    if (key == "hermite_ew") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100}, "hermite_ew");
    }
    if (key == "windowed_rank") {
        return std::make_unique<WindowedRankEstimator>(500);
    }
    return nullptr;
}
