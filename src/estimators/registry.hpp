#pragma once

#include <algorithm>
#include <cstdio>

#include <memory>
#include <string>
#include <vector>

#include "copula_family.hpp"
#include "dependence/copula_selector.hpp"
#include "dependence/estimator.hpp"
#include "dependence/estimator_bank.hpp"
#include "dependence/forgetting_bank.hpp"
#include "dependence/gaussian_mle.hpp"
#include "dependence/hermite_cusum.hpp"
#include "dependence/hermiter.hpp"
#include "dependence/kalman_bank.hpp"
#include "dependence/tail_cusum.hpp"
#include "dependence/kalman_bocpd.hpp"
#include "dependence/kalman_cusum.hpp"
#include "dependence/kalman_ewma_chart.hpp"
#include "dependence/kalman_glr.hpp"
#include "dependence/kalman_local_level.hpp"
#include "dependence/kalman_shiryaev_roberts.hpp"
#include "dependence/kalman_var_chart.hpp"
#include "dependence/mle_score_cusum.hpp"
#include "dependence/rank_score_cusum.hpp"
#include "dependence/parametric_copula.hpp"
#include "dependence/nonlinear_kalman.hpp"
#include "dependence/particle_filter.hpp"
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
    if (key == "xiao_pit") {
        // Matched to xiao_ew in memory (lambda) and bins; only the marginal
        // handling differs, so the pair isolates rank-PIT. Section 14.19.
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{
                .bins = 20, .window = 0, .lambda = 0.99, .rank_pit_window = 500},
            "xiao_pit");
    }
    if (key == "xiao_b8") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 8, .window = 0, .lambda = 0.0, .calibration = 100},
            "xiao_b8");
    }
    if (key == "xiao_b40") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 40, .window = 0, .lambda = 0.0, .calibration = 100},
            "xiao_b40");
    }
    if (key == "hermite") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.0, .burn_in = 50}, "hermite");
    }
    if (key == "hermite_ew") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.01, .burn_in = 50}, "hermite_ew");
    }
    if (key == "hermite_ew20") {
        // Control for hermite_pit: same order and burn-in, standardized marginals.
        // hermite_ew differs in BOTH order and marginal, so it cannot isolate rank-PIT.
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100}, "hermite_ew20");
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
    if (key == "kalman_cusum_noreset") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 0.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_noreset");
    }
    if (key == "kalman_cusum_soft") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 20.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_soft");
    }
    if (key == "kalman_cusum_inf2") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 2.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_inf2");
    }
    if (key == "kalman_cusum_inf5") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 5.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_inf5");
    }
    if (key == "kalman_cusum_inf10") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 10.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_inf10");
    }
    if (key == "kalman_cusum_inf50") {
        return std::make_unique<KalmanCusumDetector>(
            KalmanCusumDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2,
                                                   .burn_in_blocks = 2, .reset_inflation = 50.0},
                .delta = 1.0, .h = 3.5},
            "kalman_cusum_inf50");
    }
    // No detector at all: inflate P_ on a fixed schedule at the rate the CUSUM
    // fires. Isolates "the alarm timing was informative" from "P_ was held higher".
    if (key == "kalman_periodic_inf") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2,
                                     .reset_inflation = 20.0, .periodic_inflate_blocks = 10},
            "kalman_periodic_inf");
    }
    // Process-noise sweep, no detector. kalman_fast is q = 1e-2, the value every
    // detector wraps. If a plain filter at some other q beats the detectors, the
    // detector comparison was being run at the wrong operating point. Section 14.8.
    if (key == "kalman_q003") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 3e-2, .burn_in_blocks = 2},
            "kalman_q003");
    }
    if (key == "kalman_q01") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-1, .burn_in_blocks = 2},
            "kalman_q01");
    }
    if (key == "kalman_q03") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 3e-1, .burn_in_blocks = 2},
            "kalman_q03");
    }
    if (key == "kalman_q1") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e0, .burn_in_blocks = 2},
            "kalman_q1");
    }
    if (key == "kalman_q10") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e1, .burn_in_blocks = 2},
            "kalman_q10");
    }
    // The MLE's score is already a standardized innovation, so this is the one
    // detector in the set that needs no statistic invented for it. The three arms
    // are the same alarm with the three responses of section 14.
    if (key == "gaussian_mle_cusum") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                 .reset_deflation = -1.0},
                .block = 20, .delta = 1.0, .h = 4.0},
            "gaussian_mle_cusum");
    }
    if (key == "gaussian_mle_cusum_noreset") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                 .reset_deflation = 0.0},
                .block = 20, .delta = 1.0, .h = 4.0},
            "gaussian_mle_cusum_noreset");
    }
    if (key == "gaussian_mle_cusum_soft") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                 .reset_deflation = 20.0},
                .block = 20, .delta = 1.0, .h = 4.0},
            "gaussian_mle_cusum_soft");
    }
    // Same detector on the CUMULATIVE MLE, where theta_ adapts at 1/n and so
    // cannot chase the change away before the CUSUM sees it.
    // Same detector, score standardized by its empirical variance instead of the
    // model information, so the threshold survives a wrong family. Section 14.12.
    if (key == "gaussian_mle_cusum_rob") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                 .reset_deflation = 20.0,
                                                 .robust_score_rate = 0.001},
                .block = 20, .delta = 1.0, .h = 4.0},
            "gaussian_mle_cusum_rob");
    }
    if (key == "gaussian_mle_cusum_rob_noreset") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                 .reset_deflation = 0.0,
                                                 .robust_score_rate = 0.001},
                .block = 20, .delta = 1.0, .h = 4.0},
            "gaussian_mle_cusum_rob_noreset");
    }
    if (key == "gaussian_mle_cum_cusum") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.0, .burn_in = 50},
                .block = 20, .delta = 1.0, .h = 6.0},
            "gaussian_mle_cum_cusum");
    }
    // Controls for the score-CUSUM's soft arm, exactly as section 14.8 did for
    // the Kalman soft reset: does the alarm TIMING matter, and is the whole
    // effect reachable by tuning lambda instead?
    if (key == "gaussian_mle_periodic_defl") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50, .reset_deflation = 20.0,
                                      .periodic_deflate_ticks = 190},
            "gaussian_mle_periodic_defl");
    }
    if (key == "gaussian_mle_l95") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.95, .burn_in = 50}, "gaussian_mle_l95");
    }
    if (key == "gaussian_mle_l98") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.98, .burn_in = 50}, "gaussian_mle_l98");
    }
    if (key == "gaussian_mle_l995") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.995, .burn_in = 50}, "gaussian_mle_l995");
    }
    if (key == "gaussian_mle_l999") {
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 0.999, .burn_in = 50}, "gaussian_mle_l999");
    }
    // The section 14.10 statistic without a likelihood: a per-tick unbiased
    // estimate of Spearman's rho, residualised against the estimator's own
    // rho-hat. Estimator-agnostic, so the same detector runs on both. 14.14.
    if (key == "xiao_cusum" || key == "xiao_cusum_noreset" ||
        key == "hermite_scusum" || key == "hermite_scusum_noreset") {
        const bool hard = (key == "xiao_cusum" || key == "hermite_scusum");
        const bool is_xiao = (key == "xiao_cusum" || key == "xiao_cusum_noreset");
        std::unique_ptr<DependenceEstimator> inner =
            is_xiao ? make_estimator("xiao_ew") : make_estimator("hermite_ew");
        return std::make_unique<RankScoreCusumDetector>(
            std::move(inner),
            RankScoreCusumDetector::Config{
                .block = 20, .delta = 1.0, .h = is_xiao ? 4.5 : 5.0,
                .response = hard ? RankScoreCusumDetector::Response::Hard
                                 : RankScoreCusumDetector::Response::None},
            key == "xiao_cusum" ? "xiao_cusum"
          : key == "xiao_cusum_noreset" ? "xiao_cusum_noreset"
          : key == "hermite_scusum" ? "hermite_scusum" : "hermite_scusum_noreset");
    }
    // A posterior over the forgetting rate rather than a guess at it. 14.16.
    if (key == "kalman_bank") {
        return std::make_unique<KalmanBank>(KalmanBank::Config{}, "kalman_bank");
    }
    // Sticky and flighty variants: how fast should the memory length itself be
    // allowed to change?
    if (key == "kalman_bank_sticky") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 0.002}, "kalman_bank_sticky");
    }
    if (key == "kalman_bank_quick") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 0.15}, "kalman_bank_quick");
    }
    // Control: the same averaging with the weights FROZEN uniform. Isolates
    // "carrying several memory lengths" from "picking between them".
    if (key == "kalman_bank_uniform") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 1.0}, "kalman_bank_uniform");
    }
    // Monitors excess tail co-occurrence over what a Gaussian at the same rho
    // would give -- the functional section 14.13 says is missing. 14.17.
    if (key == "tail_cusum" || key == "tail_cusum_slow") {
        const bool slow = (key == "tail_cusum_slow");
        return std::make_unique<TailCusumDetector>(
            make_estimator("xiao_ew"),
            TailCusumDetector::Config{.count_rate = slow ? 0.0005 : 0.002, .L = 4.0},
            slow ? "tail_cusum_slow" : "tail_cusum");
    }
    // Banks over five fixed memories, M = 39, 99, 199, 399, 1999 ticks.
    //
    // ⚠️ lambda is a MIRROR-IMAGE knob (see the header of this file): for Xiao and
    // the MLE it is RETENTION, so the grid runs 0.95 (short) -> 0.999 (long); for
    // Hermite it is the UPDATE WEIGHT, so its grid runs the other way, 0.05
    // (short) -> 0.001 (long). Both grids span the same memory range. The thesis
    // writes every one of them as the update weight 2/(M+1).
    //
    // xiao_bank / hermite_bank / mle_bank and their _log twins are thesis
    // Algorithm 1 (ForgettingBank). The mle components are online MPL with
    // expected Fisher scoring (sections 3.5.4, 4.3.2). eta and delta were
    // tuned on seeds 1-30, drift and jump (notes/studies/thesis_bank_tuning.md).
    // *_bank_own scores each component by its own predictive density (Loss::Own;
    // notes/studies/thesis_bank_own_density.md); eta 0.25, delta 3e-3, tuned on seeds 1-30 for
    // the excess log loss of the bank's mixture density, not for RMSE.
    // *_bank_sq32 is the squared bank at the theory's eta = 1/32 (section 5.3.4), delta as *_bank;
    // notes/studies/thesis_bank_sq32.md.
    // *_bank_uniform freezes the weights (eta = delta = 0): the "average the arms"
    // control. *_bank_block is the earlier EstimatorBank
    // (20-tick blocks, eta_t = 1/(2 v_t)), kept so adaptive_memory.md reproduces.
    {
        const auto ends_with = [&](const char* suffix) {
            const std::string sfx(suffix);
            return key.size() > sfx.size() && key.compare(key.size() - sfx.size(), sfx.size(), sfx) == 0;
        };
        const char* base = nullptr;
        if (key == "xiao_bank" || key == "xiao_bank_log" || key == "xiao_bank_own" || key == "xiao_bank_sq32" || key == "xiao_bank_uniform"
            || key == "xiao_bank_block") base = "xiao";
        if (key == "hermite_bank" || key == "hermite_bank_log" || key == "hermite_bank_own" || key == "hermite_bank_sq32"
            || key == "hermite_bank_uniform" || key == "hermite_bank_block") base = "hermite";
        if (key == "mle_bank" || key == "mle_bank_log" || key == "mle_bank_own" || key == "mle_bank_sq32" || key == "mle_bank_uniform"
            || key == "mle_bank_block") base = "mle";
        if (base) {
            const std::string b(base);
            const bool block = ends_with("_block");
            std::vector<std::unique_ptr<DependenceEstimator>> models;
            if (b == "xiao") {
                for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                    models.push_back(std::make_unique<XiaoSpearmanEstimator>(
                        XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = lam,
                                                      .calibration = 100}, "m"));
            } else if (b == "hermite") {
                for (double lam : {0.05, 0.02, 0.01, 0.005, 0.001})
                    models.push_back(std::make_unique<HermiteEstimator>(
                        HermiteEstimator::Config{.order = 10, .lambda = lam, .burn_in = 50}, "m"));
            } else if (block) {
                for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                    models.push_back(std::make_unique<GaussianCopulaMLE>(
                        GaussianCopulaMLE::Config{.lambda = lam, .burn_in = 50}, "m"));
            } else {
                for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                    models.push_back(std::make_unique<ParametricCopulaMLE>(
                        ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .lambda = lam,
                                                    .expected_info = true, .burn_in = 50}, "m"));
            }
            static const std::vector<std::string> names = {
                "xiao_bank", "xiao_bank_log", "xiao_bank_own", "xiao_bank_sq32", "xiao_bank_uniform", "xiao_bank_block",
                "hermite_bank", "hermite_bank_log", "hermite_bank_own", "hermite_bank_sq32", "hermite_bank_uniform",
                "hermite_bank_block",
                "mle_bank", "mle_bank_log", "mle_bank_own", "mle_bank_sq32", "mle_bank_uniform", "mle_bank_block"};
            const char* name = std::find(names.begin(), names.end(), key)->c_str();
            if (block)
                return std::make_unique<EstimatorBank>(std::move(models), EstimatorBank::Config{}, name);
            ForgettingBank::Config cfg{.eta = 1.0, .delta = 1.0e-4};
            if (ends_with("_log")) {
                cfg.loss = ForgettingBank::Loss::Log;
                cfg.delta = 3.0e-4;
            } else if (ends_with("_own")) {
                cfg.loss = ForgettingBank::Loss::Own;
                cfg.eta = 0.25;
                cfg.delta = 3.0e-3;
            } else if (ends_with("_sq32")) {
                cfg.eta = 1.0 / 32.0;  // largest eta at which squared loss is exp-concave (section 5.3.4)
            } else if (ends_with("_uniform")) {
                cfg.eta = 0.0;
                cfg.delta = 0.0;
            }
            return std::make_unique<ForgettingBank>(std::move(models), cfg, name);
        }
    }
    // Online MPL (thesis section 4.3.2): sequential rank pseudo-observations into
    // expected-Fisher scoring (section 3.5.4), one per Table 3.1 family we have.
    if (key == "mpl_gaussian" || key == "mpl_clayton" || key == "mpl_gumbel" || key == "mpl_t") {
        const copula::Family f = key == "mpl_clayton" ? copula::Family::Clayton
                               : key == "mpl_gumbel"  ? copula::Family::Gumbel
                               : key == "mpl_t"       ? copula::Family::StudentT
                                                      : copula::Family::Gaussian;
        static const std::vector<std::string> names = {"mpl_gaussian", "mpl_clayton", "mpl_gumbel", "mpl_t"};
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = f, .nu = 4.0, .lambda = 0.99, .expected_info = true,
                                        .rank_pit_window = 500, .burn_in = 50},
            std::find(names.begin(), names.end(), key)->c_str());
    }
    // Resolution sweep, to test WHY Xiao beats Hermite on a singular copula
    // (section 14.15.3). If the mechanism is truncation of a smooth basis, raising
    // Hermite's order must close the gap and changing Xiao's bin count must not.
    if (key == "hermite_o20" || key == "hermite_o40") {
        const int ord = (key == "hermite_o20") ? 20 : 40;
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = ord, .lambda = 0.0, .burn_in = 50},
            key == "hermite_o20" ? "hermite_o20" : "hermite_o40");
    }
    if (key == "xiao_b8" || key == "xiao_b40") {
        const std::size_t b = (key == "xiao_b8") ? 8 : 40;
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = b, .window = 0, .lambda = 0.0,
                                          .calibration = 100},
            key == "xiao_b8" ? "xiao_b8" : "xiao_b40");
    }
    if (key == "kalman_glr") {
        return std::make_unique<KalmanGlrDetector>(
            KalmanGlrDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .window = 30, .min_lag = 1, .h = 10.0},
            "kalman_glr");
    }
    if (key == "hermite_pit") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                     .rank_pit_window = 500},
            "hermite_pit");
    }
    if (key == "hermite_vff") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                     .vff_rate = 0.5},
            "hermite_vff");
    }
    // ⚠️ MIRROR IMAGE: Xiao's lambda is RETENTION, so the VFF drives alpha =
    // 1 - lambda upward on surprise. notes/studies/adaptive_memory.md.
    if (key == "xiao_vff") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.98742,
                                          .calibration = 100, .vff_rate = 0.5},
            "xiao_vff");
    }
    if (key == "gaussian_mle_vff") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .vff_rate = 0.05, .burn_in = 50},
            "gaussian_mle_vff");
    }
    if (key == "gaussian_mle_best") {
        // Expected information + damped step. Polyak-Ruppert deliberately omitted:
        // it assumes convergence to a fixed point, which a tracking estimator never
        // does. See the ablation in section 13.
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .expected_info = true,
                                        .damped = true, .burn_in = 50},
            "gaussian_mle_best");
    }
    if (key == "gaussian_mle_einfo") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .expected_info = true, .burn_in = 50},
            "gaussian_mle_einfo");
    }
    if (key == "gaussian_mle_pr") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .pr_rate = 0.02, .burn_in = 50},
            "gaussian_mle_pr");
    }
    if (key == "gaussian_mle_damp") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .damped = true, .burn_in = 50},
            "gaussian_mle_damp");
    }
    if (key == "gaussian_mle_fisher") {
        // The three efficiency ideas together, against gaussian_mle_param as control.
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .expected_info = true,
                                        .pr_rate = 0.02, .damped = true, .burn_in = 50},
            "gaussian_mle_fisher");
    }
    if (key == "gaussian_mle_param") {
        // Control: same class as gaussian_mle_pit, no rank-PIT. Isolates whether a
        // fault is in the marginals or in ParametricCopulaMLE's Gaussian path,
        // which no registered key had exercised before.
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .burn_in = 50},
            "gaussian_mle_param");
    }
    if (key == "gaussian_mle_pit") {
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .rank_pit_window = 500, .burn_in = 50},
            "gaussian_mle_pit");
    }
    if (key == "kalman_mid") {
        // Matched sigma_omega2 for the EKF/UKF comparison: same process noise,
        // the only difference being the Fisher transform vs a nonlinear observation.
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-3, .burn_in_blocks = 2},
            "kalman_mid");
    }
    if (key == "ekf") {
        return std::make_unique<NonlinearKalman>(
            NonlinearKalman::Config{.block = 20, .sigma_omega2 = 1e-3,
                                    .mode = NonlinearKalman::Mode::Extended},
            "ekf");
    }
    if (key == "ukf") {
        return std::make_unique<NonlinearKalman>(
            NonlinearKalman::Config{.block = 20, .sigma_omega2 = 1e-3,
                                    .mode = NonlinearKalman::Mode::Unscented},
            "ukf");
    }
    if (key == "particle") {
        return std::make_unique<ParticleFilter>(
            ParticleFilter::Config{.block = 20, .particles = 400, .sigma_omega2 = 1e-4,
                                   .jump_prob = 0.01, .jump_scale = 0.5, .nu = 5.0},
            "particle");
    }
    if (key == "particle_fast") {
        return std::make_unique<ParticleFilter>(
            ParticleFilter::Config{.block = 20, .particles = 400, .sigma_omega2 = 1e-2,
                                   .jump_prob = 0.05, .jump_scale = 0.5, .nu = 5.0},
            "particle_fast");
    }
    if (key == "kalman_vff") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-3, .burn_in_blocks = 2,
                                     .vff_rate = 0.05},
            "kalman_vff");
    }
    if (key == "hermite_cusum") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                    .reset_boost = -1.0},
                .delta = 1.0, .h = 3.0, .block = 20},
            "hermite_cusum");
    }
    if (key == "hermite_cusum_noreset") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                    .reset_boost = 0.0},
                .delta = 1.0, .h = 3.0, .block = 20},
            "hermite_cusum_noreset");
    }
    if (key == "hermite_cusum_soft") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                    .reset_boost = 20.0},
                .delta = 1.0, .h = 3.0, .block = 20},
            "hermite_cusum_soft");
    }
    // The bug this replaces: baseline on the coefficient weight, so it chased
    // the change as fast as the coefficients absorbed it. Kept as a control.
    // Two-timescale statistic: a fast EWMA of the coefficient matrix against the
    // estimator's own, so per-observation noise cancels and only a persistent
    // drift of the coefficients survives. Section 14.11.
    if (key == "hermite_cusum2") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                   .surprise_fast_rate = 0.1},
                .delta = 1.0, .h = 10.0, .block = 20},
            "hermite_cusum2");
    }
    if (key == "hermite_cusum2_noreset") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                   .surprise_fast_rate = 0.1, .reset_boost = 0.0},
                .delta = 1.0, .h = 10.0, .block = 20},
            "hermite_cusum2_noreset");
    }
    if (key == "hermite_cusum_chasing") {
        return std::make_unique<HermiteCusumDetector>(
            HermiteCusumDetector::Config{
                .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                   .surprise_baseline_rate = 0.01},
                .delta = 1.0, .h = 3.0, .block = 20},
            "hermite_cusum_chasing");
    }
    if (key == "kalman_ewma_chart") {
        return std::make_unique<KalmanEwmaChartDetector>(
            KalmanEwmaChartDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .r = 0.2, .L = 2.5},
            "kalman_ewma_chart");
    }
    // Dispersion, not level: every other Kalman detector watches E[e~], this one
    // watches E[e~^2]. The filter is deliberately the fixed-memory kalman_fast --
    // a VFF filter would shorten its own memory and whiten the signal away. 14.20.
    if (key == "kalman_var_chart") {
        return std::make_unique<KalmanVarChartDetector>(
            KalmanVarChartDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .r = 0.05, .h = 1.6},
            "kalman_var_chart");
    }
    if (key == "kalman_var_chart_slow") {
        return std::make_unique<KalmanVarChartDetector>(
            KalmanVarChartDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .r = 0.02, .h = 1.225},
            "kalman_var_chart_slow");
    }
    // Same quantity, accumulated instead of averaged. k sits between the
    // worst-case null level of e~^2 (0.867, at rho=0.8) and the post-change one
    // (0.911, at rho=0.6) -- a gap the rho-dependence of the null nearly fills
    // on its own, which is the point of the measurement. 14.20.
    if (key == "kalman_var_cusum") {
        return std::make_unique<KalmanVarChartDetector>(
            KalmanVarChartDetector::Config{
                .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
                .h = 35.0, .cusum_k = 0.89},
            "kalman_var_cusum");
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
        "xiao", "xiao_ew", "xiao_win", "xiao_pit", "xiao_b8", "xiao_b40", "hermite", "hermite_ew", "hermite_ew20",
        "windowed_rank", "gaussian_mle", "gaussian_mle_ew",
        "kalman", "kalman_fast", "kalman_cusum", "kalman_cusum_noreset", "kalman_cusum_soft",
        "kalman_cusum_inf2", "kalman_cusum_inf5", "kalman_cusum_inf10", "kalman_cusum_inf50",
        "gaussian_mle_cusum", "gaussian_mle_cusum_noreset", "gaussian_mle_cusum_soft",
        "xiao_cusum", "xiao_cusum_noreset", "hermite_scusum", "hermite_scusum_noreset",
        "gaussian_mle_cusum_rob", "gaussian_mle_cusum_rob_noreset", "gaussian_mle_cum_cusum", "gaussian_mle_periodic_defl",
        "gaussian_mle_l95", "gaussian_mle_l98", "gaussian_mle_l995", "gaussian_mle_l999", "kalman_periodic_inf", "kalman_q003", "kalman_q01", "kalman_q03", "kalman_q1", "kalman_q10", "kalman_glr",
        "kalman_ewma_chart", "kalman_var_chart", "kalman_var_chart_slow", "kalman_var_cusum", "kalman_sr", "hermite_cusum", "hermite_cusum_noreset", "hermite_cusum_soft", "hermite_cusum_chasing", "hermite_cusum2", "hermite_cusum2_noreset", "hermite_vff", "xiao_vff", "hermite_pit", "gaussian_mle_vff", "kalman_vff", "particle", "particle_fast", "gaussian_mle_param", "gaussian_mle_fisher", "gaussian_mle_best", "gaussian_mle_einfo", "gaussian_mle_pr", "gaussian_mle_damp", "gaussian_mle_pit", "kalman_mid", "ekf", "ukf",
        "hermite_o20", "hermite_o40", "xiao_b8", "xiao_b40", "xiao_bank", "hermite_bank", "mle_bank", "xiao_bank_log", "hermite_bank_log", "mle_bank_log", "xiao_bank_own", "hermite_bank_own", "mle_bank_own", "xiao_bank_sq32", "hermite_bank_sq32", "mle_bank_sq32", "xiao_bank_uniform", "hermite_bank_uniform", "mle_bank_uniform", "xiao_bank_block", "hermite_bank_block", "mle_bank_block", "mpl_gaussian", "mpl_clayton", "mpl_gumbel", "mpl_t", "tail_cusum", "tail_cusum_slow", "kalman_bank", "kalman_bank_sticky", "kalman_bank_quick", "kalman_bank_uniform",
        "kalman_bocpd", "kalman_bocpd_fast",
        "clayton_mle", "clayton_mle_ew",
        "gumbel_mle", "gumbel_mle_ew",
        "t_mle", "t_mle_ew",
        "copula_select", "copula_select_ew"};
    return keys;
}

// Every key in all_estimator_keys() must construct. A key listed without a
// make_estimator branch returns nullptr and segfaults on first use, with no
// diagnostic; that has happened three times. The reverse direction — a branch
// with no list entry, so the estimator silently never runs — is checked by
// `make check-registry`.
inline bool check_registry() {
    bool ok = true;
    for (const std::string& k : all_estimator_keys()) {
        if (!make_estimator(k)) {
            std::fprintf(stderr, "registry: '%s' is in all_estimator_keys() but has no "
                                 "make_estimator branch\n", k.c_str());
            ok = false;
        }
    }
    return ok;
}
