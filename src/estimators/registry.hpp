#pragma once

#include <memory>
#include <string>
#include <vector>

#include "copula_family.hpp"
#include "dependence/copula_selector.hpp"
#include "dependence/estimator.hpp"
#include "dependence/gaussian_mle.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/kalman_bocpd.hpp"
#include "dependence/kalman_cusum.hpp"
#include "dependence/kalman_ewma_chart.hpp"
#include "dependence/kalman_glr.hpp"
#include "dependence/kalman_local_level.hpp"
#include "dependence/kalman_shiryaev_roberts.hpp"
#include "dependence/parametric_copula.hpp"
#include "dependence/windowed_rank.hpp"
#include "dependence/xiao_spearman.hpp"

// Named constructors for the nonparametric dependence estimators so experiments
// can be configured by string instead of recompiled. Returns nullptr for an
// unknown key.
//
// These configs are the single source of truth and are kept IDENTICAL to the
// live consumer's (src/consumer/), so offline and online RMSE numbers are
// directly comparable. Keep the two in sync.
//
// Forgetting conventions differ per estimator (same memory, mirror-image knob):
//   * Xiao    lambda = retention;    alpha = 1 - lambda. LARGER = longer memory.
//   * Hermite lambda = update weight; alpha = lambda.    SMALLER = longer memory.
//   * Parametric copula MLE (gaussian/clayton/gumbel/t) and the selector follow
//     Xiao: lambda = retention on the accumulated (Fisher) information.
// All then have ESS ~ (2-alpha)/alpha, so xiao lambda=0.99 and hermite
// lambda=0.01 are the same memory (ESS ~ 200), not 200 vs 100.
// For all, lambda = 0 means "cumulative" (no forgetting).
//
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
    if (key == "kalman_ewma_chart") {
        return std::make_unique<KalmanEwmaChartDetector>(
            KalmanEwmaChartDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .r = 0.2, .L = 3.0},
            "kalman_ewma_chart");
    }
    if (key == "kalman_sr") {
        return std::make_unique<KalmanShiryaevRobertsDetector>(
            KalmanShiryaevRobertsDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .delta = 1.0, .h = 5.0},
            "kalman_sr");
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

// Every key above, in report order. The harnesses share this rather than each
// keeping a copy: a key registered but missing from a harness's list is not an
// error, just a silently absent row, which is how xiao_win went unmeasured.
inline const std::vector<std::string>& all_estimator_keys() {
    static const std::vector<std::string> keys = {
        "xiao", "xiao_ew", "xiao_win", "hermite", "hermite_ew",
        "windowed_rank", "gaussian_mle", "gaussian_mle_ew",
        "kalman", "kalman_fast", "kalman_cusum", "kalman_glr",
        "kalman_ewma_chart", "kalman_sr",
        "kalman_bocpd", "kalman_bocpd_fast",
        "clayton_mle", "clayton_mle_ew",
        "gumbel_mle", "gumbel_mle_ew",
        "t_mle", "t_mle_ew",
        "copula_select", "copula_select_ew"};
    return keys;
}
