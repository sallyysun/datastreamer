#pragma once

#include <memory>
#include <string>

#include "dependence/estimator.hpp"
#include "dependence/gaussian_mle.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/kalman_bocpd.hpp"
#include "dependence/kalman_cusum.hpp"
#include "dependence/kalman_glr.hpp"
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
//   * Xiao    lambda = retention;    alpha = 1 - lambda. LARGER = longer memory.
//   * Hermite lambda = update weight; alpha = lambda.    SMALLER = longer memory.
// Both then have ESS ~ (2-alpha)/alpha, so xiao lambda=0.99 and hermite
// lambda=0.01 are the same memory (ESS ~ 200), not 200 vs 100.
// For both, lambda = 0 means "cumulative" (no forgetting).
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
    // Forgetting knob is sigma_omega2. `block` trades lag against observation
    // noise and must stay under the fastest timescale in the DGP suite.
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
    // Detector thresholds below are calibrated by calibrate_arl0.cpp; re-run it
    // if block, sigma_omega2 or hazard_rate changes.
    if (key == "kalman_cusum") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum");
    }
    if (key == "kalman_glr") {
        return std::make_unique<KalmanGlrDetector>(
            KalmanGlrDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .window = 30, .min_lag = 1, .h = 10.0},
            "kalman_glr");
    }
    if (key == "kalman_bocpd") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 1000.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.15},
            "kalman_bocpd");
    }
    // hazard_rate is BOCPD's analog of sigma_omega2; the threshold rises with it
    // because a model expecting frequent change finds short runs likelier.
    if (key == "kalman_bocpd_fast") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 10.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.95},
            "kalman_bocpd_fast");
    }
    return nullptr;
}
