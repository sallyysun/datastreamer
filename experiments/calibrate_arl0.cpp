// ARL0 calibration for KalmanCusumDetector.
//
// change_detection.md Part VI is explicit that no detector number is a thesis
// result until it is compared at a MATCHED in-control average run length
// (ARL0), and that ARL0 must be bootstrapped on the ACTUAL pseudo-observation
// stream -- not read off a classical CUSUM formula, since rank/PIT feedback
// breaks the iid assumption those formulas need. This is that bootstrap: run
// KalmanCusumDetector's real update() path on a stationary Gaussian-copula
// stream at several rho, sweep the threshold h, and report the smallest h
// that meets a target ARL0 in the WORST case over rho (since the true rho is
// unknown in live use, the calibrated threshold must be safe across the
// range this project's DGP suite actually exercises).
//
// Usage: ./calibrate_arl0 [target_arl0_blocks]   (default 1000)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../src/estimators/dependence/kalman_cusum.hpp"

namespace {

// Mean blocks-between-alarms at threshold h, pooled over several independent
// seeds at a fixed simulation horizon. Negative return = no alarms were
// observed at all -- a LOWER BOUND on ARL0, not a point estimate (conflating
// the two was a bug caught in the scratch version of this check).
double measure_arl0(double h, double rho, long ticks_per_seed, int n_seeds, unsigned seed0) {
    long total_alarms = 0;
    for (int s = 0; s < n_seeds; ++s) {
        KalmanCusumDetector::Config cfg{
            .kalman = KalmanLocalLevel::Config{.block = 20, .sigma_omega2 = 1e-2, .burn_in_blocks = 2},
            .delta = 1.0, .h = h};
        KalmanCusumDetector det(cfg, "kalman_cusum");
        std::mt19937 rng(seed0 + static_cast<unsigned>(s));
        std::normal_distribution<double> z(0.0, 1.0);
        for (long t = 0; t < ticks_per_seed; ++t) {
            const double zx = z(rng);
            const double zy = rho * zx + std::sqrt(std::max(0.0, 1.0 - rho * rho)) * z(rng);
            det.update(zx, zy);
            if (det.alarmed_this_block()) ++total_alarms;
        }
    }
    const double total_blocks = (static_cast<double>(ticks_per_seed) / 20.0) * n_seeds;
    return total_alarms > 0 ? total_blocks / static_cast<double>(total_alarms) : -total_blocks;
}

}  // namespace

int main(int argc, char** argv) {
    const double target = (argc > 1) ? std::atof(argv[1]) : 1000.0;
    const long ticks_per_seed = 100000;  // 5000 blocks/seed at block=20
    const int n_seeds = 6;
    const std::vector<double> rhos = {0.0, 0.3, 0.6, 0.8};  // spans the DGP suite's stationary levels

    std::printf("=== ARL0(h) sweep, worst case over rho in {0.0, 0.3, 0.6, 0.8} ===\n");
    std::printf("(%d seeds x %ld ticks = %.0f blocks per (h, rho) cell; target ARL0 = %.0f blocks)\n\n",
                 n_seeds, ticks_per_seed, (static_cast<double>(ticks_per_seed) / 20.0) * n_seeds, target);

    const std::vector<double> hs = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    double calibrated_h = -1.0;
    for (double h : hs) {
        double worst_arl0 = 1e18;
        double worst_rho = rhos[0];
        for (double rho : rhos) {
            const double arl0 = measure_arl0(h, rho, ticks_per_seed, n_seeds, 1000);
            const double reported = std::fabs(arl0);
            std::printf("  h=%.1f  rho=%.1f  ARL0%s=%.0f blocks\n",
                        h, rho, arl0 < 0 ? " (lower bound, zero alarms observed)" : "", reported);
            if (reported < worst_arl0) { worst_arl0 = reported; worst_rho = rho; }
        }
        const bool meets = worst_arl0 >= target;
        std::printf("h=%.1f  worst-case (rho=%.1f) ARL0 = %.0f blocks  -- %s\n\n",
                     h, worst_rho, worst_arl0, meets ? "meets target" : "BELOW target");
        if (meets && calibrated_h < 0.0) calibrated_h = h;
    }

    if (calibrated_h < 0.0) {
        std::printf("no h in the sweep range met the target -- widen the sweep\n");
        return 1;
    }
    std::printf("calibrated h = %.1f for target ARL0 = %.0f blocks (worst case over rho)\n",
                calibrated_h, target);
    return 0;
}
