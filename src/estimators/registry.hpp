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
    // Innovation CUSUM on kalman_fast's standardized innovations, detect-and-
    // restart via reset() on alarm (change_detection.md Part III Level 1).
    // h=3.5 is CALIBRATED, not a classical-table default: experiments/
    // calibrate_arl0.cpp bootstraps ARL0 directly on this pipeline's real
    // block-Spearman innovations (not synthetic N(0,1) draws, per Part VI) at
    // several stationary rho, and picks the smallest h whose WORST CASE over
    // rho in {0.0, 0.3, 0.6, 0.8} still clears a target ARL0 of 1000 blocks
    // (worst case: rho=0.6, empirical ARL0=2143 blocks -- 2x the target
    // margin). The earlier uncalibrated default (h=5) turned out to be far
    // more conservative than needed: its own measured ARL0 is >=30000 blocks,
    // i.e. it was trading away detection speed for a false-alarm budget
    // nobody asked for. Re-run calibrate_arl0.cpp if block or sigma_omega2
    // changes -- ARL0 depends on both.
    if (key == "kalman_cusum") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum");
    }
    // Window-limited GLR (change_detection.md Part III Level 2): the honest
    // version of kalman_cusum for an UNKNOWN jump size, built as a matched
    // filter on the filter's own closed-form step response
    // rho_s(tau)=(1-K)^{s-tau} -- Monte-Carlo verified against this exact
    // filter (empirical decay ratio 0.65-0.74 per block vs theoretical
    // (1-K)=0.672 at sigma_omega2=1e-2, block=20). window=30 blocks (600
    // ticks) bounds it to O(window) per block, per Lai's window-limited
    // construction -- the unbounded sup-over-tau GLR is not
    // streaming-admissible. h=10.0 is CALIBRATED the same way as
    // kalman_cusum's h (experiments/calibrate_arl0.cpp, same target ARL0 of
    // 1000 blocks): worst case over rho in {0.0, 0.3, 0.6, 0.8} is
    // rho=0.8, empirical ARL0=2727 blocks (2.7x margin).
    if (key == "kalman_glr") {
        return std::make_unique<KalmanGlrDetector>(
            KalmanGlrDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .window = 30, .min_lag = 1, .h = 10.0},
            "kalman_glr");
    }
    // Bayesian Online Changepoint Detection (change_detection.md Part III
    // Level 3 / build order step 6) on the same blockwise Fisher-z'd
    // Spearman observation as the estimators above. Unlike kalman_cusum and
    // kalman_glr, there is no reset() call anywhere -- estimate() is the
    // run-length-weighted posterior mean, a soft Bayesian model average
    // rather than a hard detect-and-restart. hazard_rate=1/1000 blocks
    // matches the ARL0=1000-block target the other two detectors were
    // calibrated against, so all three share the same "expected time between
    // changes" operating point. alarm_prob_threshold=0.15 is separately
    // ARL0-calibrated (experiments/calibrate_arl0.cpp) against the SAME
    // target: worst case over rho in {0.0, 0.3, 0.6, 0.8} is rho=0.0,
    // empirical ARL0=1071 blocks (only a 7% margin -- thinner than
    // kalman_cusum/kalman_glr's 2x/2.7x, since the sweep grid here is coarser
    // near the boundary; tighten the grid before trusting this margin
    // closely).
    if (key == "kalman_bocpd") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 1000.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.15},
            "kalman_bocpd");
    }
    return nullptr;
}
