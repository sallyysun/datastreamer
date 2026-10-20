#pragma once

// Memory-grid estimator construction shared by the async harnesses.
// See notes/studies/async_memory.md.

#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/gaussian_mle.hpp"
#include "../src/estimators/dependence/hermiter.hpp"
#include "../src/estimators/dependence/kalman_local_level.hpp"
#include "../src/estimators/dependence/windowed_rank.hpp"
#include "../src/estimators/dependence/xiao_spearman.hpp"

namespace memgrid {

constexpr std::size_t KALMAN_BLOCK = 20;
inline const std::vector<std::string> kFamilies = {"xiao_ew", "hermite_ew", "gaussian_mle_ew", "windowed_rank", "kalman"};

// Memory M = effective sample size in pairs; every EW knob has ESS = (2-a)/a.
// Returns nullptr where the family cannot express M (Kalman below 2 blocks).
inline std::unique_ptr<DependenceEstimator> make_at_memory(const std::string& family, double M) {
    const double a = 2.0 / (M + 1.0);
    if (family == "xiao_ew")
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 1.0 - a, .calibration = 100}, "xiao_ew");
    if (family == "hermite_ew")
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = a, .burn_in = 50}, "hermite_ew");
    if (family == "gaussian_mle_ew")
        return std::make_unique<GaussianCopulaMLE>(
            GaussianCopulaMLE::Config{.lambda = 1.0 - a, .burn_in = 50}, "gaussian_mle_ew");
    if (family == "windowed_rank")
        return std::make_unique<WindowedRankEstimator>(static_cast<std::size_t>(M));
    if (family == "kalman") {
        // Steady-state gain K = 1 - lambda gives ESS (2-K)/K blocks; invert for sigma_omega2.
        const double Mb = M / static_cast<double>(KALMAN_BLOCK);
        if (Mb < 2.0) return nullptr;
        const double K = 2.0 / (Mb + 1.0);
        const double r = 1.06 / (static_cast<double>(KALMAN_BLOCK) - 3.0);
        return std::make_unique<KalmanLocalLevel>(
            KalmanLocalLevel::Config{.block = KALMAN_BLOCK, .sigma_omega2 = K * K * r / (1.0 - K), .burn_in_blocks = 2},
            "kalman");
    }
    return nullptr;
}

}  // namespace memgrid
