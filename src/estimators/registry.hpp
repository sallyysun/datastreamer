#pragma once

#include <memory>
#include <string>

#include "dependence/estimator.hpp"
#include "dependence/gaussian_mle.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/kalman_local_level.hpp"
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
//   For both, lambda = 0 means "cumulative" (no forgetting).
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
    // Kalman local-level filter (change_detection.md Bridge 1): a strict
    // generalization of the fixed-lambda estimators above, not a sibling of
    // them -- the free knob is sigma_omega2 (process noise, "how fast does
    // rho actually wander"), not a hand-picked lambda. Two operating points
    // for now, same slow/fast contrast as the *_ew pairs elsewhere; VFF
    // (estimating sigma_omega2 online) is future work.
    //
    // `block` is a second, less obvious knob: the observation model treats
    // rho as constant WITHIN a block, so block must stay well under the
    // fastest true timescale in the DGP suite or that assumption breaks and
    // the filter reports a materially biased read, not just a noisier one.
    // Swept empirically against `oscillate` (period 400, the fastest scenario
    // here): block=50 (1/8 period) already costs ~35% extra RMSE relative to
    // block=20 at matched sigma_omega2; block=10 keeps improving oscillate but
    // starts costing accuracy on `constant` (noisier per-block Fisher-z read,
    // sigma_v2 ~ 1.06/(block-3)). block=20 is the harness-wide sweet spot.
    if (key == "kalman") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-4, .burn_in_blocks = 2},
            "kalman");
    }
    if (key == "kalman_fast") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
            "kalman_fast");
    }
    return nullptr;
}
