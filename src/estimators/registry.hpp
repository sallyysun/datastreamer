#pragma once

#include <cstdio>

#include <memory>
#include <string>
#include <vector>

#include "copula_family.hpp"
#include "dependence/copula_selector.hpp"
#include "dependence/estimator.hpp"
#include "dependence/estimator_bank.hpp"
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

// Estimator configs by string key (nullptr if unknown). Results per key:
// notes/studies/kalman_changepoint_results.md. Keep IDENTICAL to src/consumer/.
// ⚠️ lambda is a mirror-image knob: Xiao, the copula MLEs and the selector use
// RETENTION (alpha = 1 - lambda, larger = longer); Hermite uses the UPDATE WEIGHT
// (alpha = lambda, smaller = longer). ESS ~ (2-alpha)/alpha, so xiao 0.99 ==
// hermite 0.01. lambda = 0 is cumulative. Clayton/Gumbel: positive dependence only.
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
        // Control pair with xiao_ew: only the marginal handling differs. §14.19.
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
    // Forgetting knob is sigma_omega2; `block` must stay under the fastest DGP timescale.
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
    // Detector thresholds are calibrated by calibrate_arl0.cpp; re-run it if
    // block, sigma_omega2 or hazard_rate changes.
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
    // Control: no detector, P_ inflated on a fixed schedule at the CUSUM's alarm rate.
    if (key == "kalman_periodic_inf") {
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2,
                                     .reset_inflation = 20.0, .periodic_inflate_blocks = 10},
            "kalman_periodic_inf");
    }
    // Process-noise sweep, no detector; every detector wraps kalman_fast (q = 1e-2). §14.8.
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
    // Score standardized by its empirical variance, so h survives a wrong family. §14.12.
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
    // Cumulative MLE: theta_ adapts at 1/n, so it cannot chase the change away first.
    if (key == "gaussian_mle_cum_cusum") {
        return std::make_unique<MleScoreCusumDetector>(
            MleScoreCusumDetector::Config{
                .mle = GaussianCopulaMLE::Config{.lambda = 0.0, .burn_in = 50},
                .block = 20, .delta = 1.0, .h = 6.0},
            "gaussian_mle_cum_cusum");
    }
    // Controls for the score-CUSUM soft arm (as §14.8): fixed-schedule deflation, lambda sweep.
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
    // Rank-residual CUSUM, estimator-agnostic. §14.14.
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
    // §14.16.
    if (key == "kalman_bank") {
        return std::make_unique<KalmanBank>(KalmanBank::Config{}, "kalman_bank");
    }
    if (key == "kalman_bank_sticky") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 0.002}, "kalman_bank_sticky");
    }
    if (key == "kalman_bank_quick") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 0.15}, "kalman_bank_quick");
    }
    // Control: weights frozen uniform.
    if (key == "kalman_bank_uniform") {
        return std::make_unique<KalmanBank>(
            KalmanBank::Config{.switch_prob = 1.0}, "kalman_bank_uniform");
    }
    // §14.17.
    if (key == "tail_cusum" || key == "tail_cusum_slow") {
        const bool slow = (key == "tail_cusum_slow");
        return std::make_unique<TailCusumDetector>(
            make_estimator("xiao_ew"),
            TailCusumDetector::Config{.count_rate = slow ? 0.0005 : 0.002, .L = 4.0},
            slow ? "tail_cusum_slow" : "tail_cusum");
    }
    // §14.18. ⚠️ Mirror-image lambda: the Hermite grid runs the opposite way over
    // the same memory range as the Xiao/MLE retention grid.
    if (key == "xiao_bank" || key == "hermite_bank" || key == "mle_bank") {
        std::vector<std::unique_ptr<DependenceEstimator>> models;
        if (key == "xiao_bank") {
            for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                models.push_back(std::make_unique<XiaoSpearmanEstimator>(
                    XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = lam,
                                                  .calibration = 100}, "m"));
        } else if (key == "hermite_bank") {
            for (double lam : {0.05, 0.02, 0.01, 0.005, 0.001})
                models.push_back(std::make_unique<HermiteEstimator>(
                    HermiteEstimator::Config{.order = 10, .lambda = lam, .burn_in = 50}, "m"));
        } else {
            for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                models.push_back(std::make_unique<GaussianCopulaMLE>(
                    GaussianCopulaMLE::Config{.lambda = lam, .burn_in = 50}, "m"));
        }
        return std::make_unique<EstimatorBank>(
            std::move(models), EstimatorBank::Config{},
            key == "xiao_bank" ? "xiao_bank" : key == "hermite_bank" ? "hermite_bank" : "mle_bank");
    }
    // Resolution sweep: Hermite order vs Xiao bins on a singular copula. §14.15.3.
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
    // ⚠️ Xiao's lambda is RETENTION: the VFF drives alpha = 1 - lambda up on surprise.
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
        // Polyak-Ruppert deliberately omitted: a tracking estimator never converges. §13.
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
        return std::make_unique<ParametricCopulaMLE>(
            ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .nu = 4.0,
                                        .lambda = 0.99, .expected_info = true,
                                        .pr_rate = 0.02, .damped = true, .burn_in = 50},
            "gaussian_mle_fisher");
    }
    if (key == "gaussian_mle_param") {
        // Control for gaussian_mle_pit and gaussian_mle_fisher: same class, no extras.
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
        // Matched sigma_omega2 for the EKF/UKF comparison.
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
    // Two-timescale surprise. §14.11.
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
    // Control: the §14.11 bug, baseline chasing the coefficient weight.
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
    // Watches E[e~^2]; fixed-memory filter on purpose (a VFF would whiten the signal). §14.20.
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
    // k between null e~^2 (0.867 at rho=0.8) and post-change (0.911 at rho=0.6). §14.20.
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
    // Threshold rises with hazard_rate: a model expecting change finds short runs likelier.
    if (key == "kalman_bocpd_fast") {
        return std::make_unique<KalmanBocpdDetector>(
            KalmanBocpdDetector::Config{
                .block = 20, .hazard_rate = 1.0 / 10.0,
                .short_run_cutoff = 2, .alarm_prob_threshold = 0.95},
            "kalman_bocpd_fast");
    }
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

// Both harnesses read this list; a key registered but missing here is silently never run.
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
        "hermite_o20", "hermite_o40", "xiao_b8", "xiao_b40", "xiao_bank", "hermite_bank", "mle_bank", "tail_cusum", "tail_cusum_slow", "kalman_bank", "kalman_bank_sticky", "kalman_bank_quick", "kalman_bank_uniform",
        "kalman_bocpd", "kalman_bocpd_fast",
        "clayton_mle", "clayton_mle_ew",
        "gumbel_mle", "gumbel_mle_ew",
        "t_mle", "t_mle_ew",
        "copula_select", "copula_select_ew"};
    return keys;
}

// A listed key with no branch returns nullptr and segfaults on first use.
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
