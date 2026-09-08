// ARL0 calibration for the Kalman-filter change detectors: sweeps each
// detector's threshold on stationary streams at several rho and reports the
// smallest one meeting the target in the worst case over rho.
// See notes/studies/kalman_changepoint_results.md.
//
// Usage: ./calibrate_arl0 [target_arl0_blocks]   (default 1000)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/dependence/change_detector.hpp"
#include "../src/estimators/dependence/kalman_bocpd.hpp"
#include "../src/estimators/dependence/kalman_cusum.hpp"
#include "../src/estimators/dependence/kalman_ewma_chart.hpp"
#include "../src/estimators/dependence/kalman_glr.hpp"
#include "../src/estimators/dependence/kalman_shiryaev_roberts.hpp"
#include "../src/estimators/dependence/mle_score_cusum.hpp"
#include "../src/estimators/dependence/rank_score_cusum.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/estimators/dependence/hermite_cusum.hpp"

namespace {

// Mean blocks-between-alarms, pooled over seeds. Negative = no alarms seen,
// i.e. a lower bound on ARL0 rather than a point estimate.
double measure_arl0(const std::function<std::unique_ptr<DependenceEstimator>()>& make_detector,
                     double rho, long ticks_per_seed, int n_seeds, unsigned seed0) {
    long total_alarms = 0;
    for (int s = 0; s < n_seeds; ++s) {
        auto est = make_detector();
        auto* det = dynamic_cast<ChangeDetector*>(est.get());
        std::mt19937 rng(seed0 + static_cast<unsigned>(s));
        std::normal_distribution<double> z(0.0, 1.0);
        for (long t = 0; t < ticks_per_seed; ++t) {
            const double zx = z(rng);
            const double zy = rho * zx + std::sqrt(std::max(0.0, 1.0 - rho * rho)) * z(rng);
            est->update(zx, zy);
            if (det->alarmed_this_block()) ++total_alarms;
        }
    }
    const double total_blocks = (static_cast<double>(ticks_per_seed) / 20.0) * n_seeds;
    return total_alarms > 0 ? total_blocks / static_cast<double>(total_alarms) : -total_blocks;
}

double calibrate(const char* label,
                  const std::function<std::unique_ptr<DependenceEstimator>(double h)>& make_detector,
                  const std::vector<double>& hs, double target,
                  long ticks_per_seed, int n_seeds, const std::vector<double>& rhos) {
    std::printf("=== %s: ARL0(h) sweep, worst case over rho in {0.0, 0.3, 0.6, 0.8} ===\n", label);
    std::printf("(%d seeds x %ld ticks = %.0f blocks per (h, rho) cell; target ARL0 = %.0f blocks)\n\n",
                 n_seeds, ticks_per_seed, (static_cast<double>(ticks_per_seed) / 20.0) * n_seeds, target);

    double calibrated_h = -1.0;
    for (double h : hs) {
        double worst_arl0 = 1e18;
        double worst_rho = rhos[0];
        for (double rho : rhos) {
            const double arl0 = measure_arl0([&make_detector, h] { return make_detector(h); },
                                              rho, ticks_per_seed, n_seeds, 1000);
            const double reported = std::fabs(arl0);
            std::printf("  h=%.2f  rho=%.2f  ARL0%s=%.0f blocks\n",
                        h, rho, arl0 < 0 ? " (lower bound, zero alarms observed)" : "", reported);
            if (reported < worst_arl0) { worst_arl0 = reported; worst_rho = rho; }
        }
        const bool meets = worst_arl0 >= target;
        std::printf("h=%.2f  worst-case (rho=%.2f) ARL0 = %.0f blocks  -- %s\n\n",
                     h, worst_rho, worst_arl0, meets ? "meets target" : "BELOW target");
        if (meets && calibrated_h < 0.0) calibrated_h = h;
    }
    if (calibrated_h < 0.0) {
        std::printf("%s: no h in the sweep range met the target -- widen the sweep\n\n", label);
        return -1.0;
    }
    std::printf("%s: calibrated h = %.2f for target ARL0 = %.0f blocks (worst case over rho)\n\n",
                label, calibrated_h, target);
    return calibrated_h;
}

}  // namespace

int main(int argc, char** argv) {
    const double target = (argc > 1) ? std::atof(argv[1]) : 1000.0;
    const long ticks_per_seed = 100000;
    const int n_seeds = 6;
    const std::vector<double> rhos = {0.0, 0.3, 0.6, 0.8};
    const KalmanLocalLevel::Config kalman_cfg{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2};

    calibrate(
        "kalman_cusum",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanCusumDetector>(
                KalmanCusumDetector::Config{.kalman = kalman_cfg, .delta = 1.0, .h = h}, "kalman_cusum");
        },
        {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "kalman_glr",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanGlrDetector>(
                KalmanGlrDetector::Config{.kalman = kalman_cfg, .window = 30, .min_lag = 1, .h = h}, "kalman_glr");
        },
        {5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "kalman_ewma_chart",
        [&](double L) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanEwmaChartDetector>(
                KalmanEwmaChartDetector::Config{.kalman = kalman_cfg, .r = 0.2, .L = L},
                "kalman_ewma_chart");
        },
        {2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "kalman_sr",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanShiryaevRobertsDetector>(
                KalmanShiryaevRobertsDetector::Config{.kalman = kalman_cfg, .delta = 1.0, .h = h},
                "kalman_sr");
        },
        {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "gaussian_mle_cusum",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<MleScoreCusumDetector>(
                MleScoreCusumDetector::Config{
                    .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50},
                    .block = 20, .delta = 1.0, .h = h},
                "gaussian_mle_cusum");
        },
        {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "gaussian_mle_cum_cusum",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<MleScoreCusumDetector>(
                MleScoreCusumDetector::Config{
                    .mle = GaussianCopulaMLE::Config{.lambda = 0.0, .burn_in = 50},
                    .block = 20, .delta = 1.0, .h = h},
                "gaussian_mle_cum_cusum");
        },
        {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "hermite_cusum",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<HermiteCusumDetector>(
                HermiteCusumDetector::Config{
                    .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100},
                    .delta = 1.0, .h = h, .block = 20},
                "hermite_cusum");
        },
        {1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "hermite_cusum2",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<HermiteCusumDetector>(
                HermiteCusumDetector::Config{
                    .hermite = HermiteEstimator::Config{.order = 20, .lambda = 0.01, .burn_in = 100,
                                                       .surprise_fast_rate = 0.1},
                    .delta = 1.0, .h = h, .block = 20},
                "hermite_cusum2");
        },
        {3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "gaussian_mle_cusum_rob",
        [&](double h) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<MleScoreCusumDetector>(
                MleScoreCusumDetector::Config{
                    .mle = GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50,
                                                     .robust_score_rate = 0.001},
                    .block = 20, .delta = 1.0, .h = h},
                "gaussian_mle_cusum_rob");
        },
        {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0}, target, ticks_per_seed, n_seeds, rhos);

    for (const char* inner : {"xiao_ew", "hermite_ew"}) {
        calibrate(
            (std::string(inner) + " rank score-CUSUM").c_str(),
            [&](double h) -> std::unique_ptr<DependenceEstimator> {
                return std::make_unique<RankScoreCusumDetector>(
                    make_estimator(inner),
                    RankScoreCusumDetector::Config{
                        .block = 20, .delta = 1.0, .h = h,
                        // ARL0 is a property of the STATISTIC, so calibrate with
                        // the response switched off; a response that feeds back
                        // would otherwise be calibrated into the threshold.
                        .response = RankScoreCusumDetector::Response::None}, "probe");
            },
            {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0}, target, ticks_per_seed, n_seeds, rhos);
    }

    // The BOCPD variants sweep alarm_prob_threshold rather than h.
    calibrate(
        "kalman_bocpd",
        [&](double thresh) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanBocpdDetector>(
                KalmanBocpdDetector::Config{.block = 20, .hazard_rate = 1.0 / 1000.0,
                                             .short_run_cutoff = 2, .alarm_prob_threshold = thresh},
                "kalman_bocpd");
        },
        {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9}, target, ticks_per_seed, n_seeds, rhos);

    calibrate(
        "kalman_bocpd_fast",
        [&](double thresh) -> std::unique_ptr<DependenceEstimator> {
            return std::make_unique<KalmanBocpdDetector>(
                KalmanBocpdDetector::Config{.block = 20, .hazard_rate = 1.0 / 10.0,
                                             .short_run_cutoff = 2, .alarm_prob_threshold = thresh},
                "kalman_bocpd_fast");
        },
        {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999}, target, ticks_per_seed, n_seeds, rhos);

    return 0;
}
