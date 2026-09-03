#pragma once

#include <memory>
#include <string>

#include "copula_family.hpp"
#include "dependence/copula_selector.hpp"
#include "dependence/estimator.hpp"
#include "dependence/gaussian_mle.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/parametric_copula.hpp"
#include "dependence/windowed_rank.hpp"
#include "dependence/xiao_spearman.hpp"

// Named constructors for the nonparametric dependence estimators so experiments
// can be configured by string instead of recompiled. Returns nullptr for an
// unknown key.
//
// These configs are the single source of truth and are kept IDENTICAL to the
// live DependenceMonitor (src/device/dependence_monitor.hpp), so offline and
// online RMSE numbers are directly comparable. Keep the two in sync.
//
// Forgetting conventions differ per estimator (same memory, mirror-image knob):
//   * Xiao    lambda = retention in (0,1); LARGER = longer memory.
//             Effective sample size ~ (1+lambda)/(1-lambda). lambda=0.99 => ESS~200.
//   * Hermite lambda = EWMA update weight in (0,1); SMALLER = longer memory.
//             Effective sample size ~ 1/lambda. lambda=0.01 => ESS~100.
//   * Parametric copula MLE (gaussian/clayton/gumbel/t) and the selector use the
//             SAME convention as Xiao: lambda = retention on the accumulated
//             (Fisher) information; LARGER = longer memory (ESS ~ 1/(1-lambda)).
//   For all, lambda = 0 means "cumulative" (no forgetting).
//
// The parametric families (Clayton/Gumbel/Student-t) and the online
// parametric-vs-nonparametric goodness-of-fit selector live in
// dependence/parametric_copula.hpp and dependence/copula_selector.hpp; the
// family math (log-density, score, Spearman conversion) is in copula_family.hpp.
// Clayton/Gumbel model POSITIVE dependence only; the t copula's nu is fixed.
inline std::unique_ptr<DependenceEstimator> make_estimator(const std::string& key) {
    if (key == "xiao") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.0, .calibration = 100},
            "xiao");
    }
    if (key == "xiao_ew") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.99, .calibration = 100},
            "xiao_ew");
    }
    if (key == "xiao_win") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 500, .lambda = 0.0, .calibration = 100},
            "xiao_win");
    }
    if (key == "hermite") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.0, .burn_in = 50}, "hermite");
    }
    if (key == "hermite_ew") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.01, .burn_in = 50}, "hermite_ew");
    }
    if (key == "windowed_rank") {
        return std::make_unique<WindowedRankEstimator>(500);
    }
    if (key == "gaussian_mle") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.0, .burn_in = 50}, "gaussian_mle");
    }
    if (key == "gaussian_mle_ew") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50}, "gaussian_mle_ew");
    }
    // --- Non-Gaussian parametric copula MLE (positive-dependence families) ---
    if (key == "clayton_mle") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Clayton, .lambda = 0.0, .burn_in = 50},
            "clayton_mle");
    }
    if (key == "clayton_mle_ew") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Clayton, .lambda = 0.99, .burn_in = 50},
            "clayton_mle_ew");
    }
    if (key == "gumbel_mle") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gumbel, .lambda = 0.0, .burn_in = 50},
            "gumbel_mle");
    }
    if (key == "gumbel_mle_ew") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gumbel, .lambda = 0.99, .burn_in = 50},
            "gumbel_mle_ew");
    }
    if (key == "t_mle") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::StudentT, .nu = 4.0, .lambda = 0.0, .burn_in = 50},
            "t_mle");
    }
    if (key == "t_mle_ew") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::StudentT, .nu = 4.0, .lambda = 0.99, .burn_in = 50},
            "t_mle_ew");
    }
    // --- Online parametric-vs-nonparametric goodness-of-fit selection ---
    if (key == "copula_select") {
        return std::make_unique<CopulaSelector>(
            CopulaSelector::Config{.lambda = 0.0, .burn_in = 100, .nu = 4.0}, "copula_select");
    }
    if (key == "copula_select_ew") {
        return std::make_unique<CopulaSelector>(
            CopulaSelector::Config{.lambda = 0.99, .burn_in = 100, .nu = 4.0}, "copula_select_ew");
    }
    return nullptr;
}
