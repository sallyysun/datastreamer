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
    // Kalman local-level filter. The forgetting knob is sigma_omega2 (process
    // noise) rather than a hand-picked lambda; the two operating points below
    // mirror the slow/fast contrast of the *_ew pairs above.
    //
    // `block` matters as much: the observation model treats rho as constant
    // within a block, so it must stay well under the fastest timescale in the
    // DGP suite or the read is biased, not merely noisier. Larger blocks also
    // cut observation noise (sigma_v2 ~ 1.06/(block-3)), so it trades lag
    // against variance; 20 is the harness-wide compromise.
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
    // Innovation CUSUM with detect-and-restart on alarm. h is calibrated by
    // experiments/calibrate_arl0.cpp against a target ARL0, not taken from a
    // classical table. Re-run it if block or sigma_omega2 changes -- ARL0
    // depends on both.
    if (key == "kalman_cusum") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum");
    }
    // Window-limited GLR: kalman_cusum's counterpart for an unknown jump size.
    // `window` bounds the candidate jump lags, keeping cost O(window) per
    // block. h is calibrated the same way as kalman_cusum's.
    if (key == "kalman_glr") {
        return std::make_unique<KalmanGlrDetector>(
            KalmanGlrDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .window = 30, .min_lag = 1, .h = 10.0},
            "kalman_glr");
    }
    // BOCPD on the same observation. Unlike the two detectors above there is
    // no reset() anywhere -- estimate() is the run-length-weighted posterior
    // mean. hazard_rate matches the ARL0 target the others were calibrated
    // against, so all of them sit at the same expected-time-between-changes
    // operating point; alarm_prob_threshold is calibrated separately.
    if (key == "kalman_bocpd") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 1000.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.15},
            "kalman_bocpd");
    }
    // Same model with hazard_rate retuned to match `oscillate`'s half-period
    // (10 blocks). Hazard is BOCPD's analog of sigma_omega2 -- its own "how
    // fast do I expect change" knob -- so no single value suits every regime.
    // The threshold has to rise with the hazard: a model that expects frequent
    // change finds "the run is short" true more often, so holding the same
    // ARL0 needs a stricter bar.
    if (key == "kalman_bocpd_fast") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 10.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.95},
            "kalman_bocpd_fast");
    }
    return nullptr;
}
