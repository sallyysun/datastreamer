#pragma once

// Multiple forgetting factors (Chapter 5): Hedge with fixed share over fixed-memory arms, given the shared
// pseudo-observations (U, V) from outside.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "../copula_family.hpp"
#include "estimator.hpp"

namespace thesis {

constexpr std::size_t kK = 5;
inline const double kMem[kK] = {39, 99, 199, 399, 1999};   // MFF arms = the fixed memories

// MFF: Hedge with fixed share over the fixed-memory arms, log loss of the Gaussian density at each arm's rho_S,
// eta = 1, delta = 1e-4; forecast = weighted mean of the arms (density: the weighted mixture).
struct Mff {
    double eta = 1.0, delta = 1e-4;
    std::vector<double> w = std::vector<double>(kK, 1.0 / kK), loss = std::vector<double>(kK), lw = std::vector<double>(kK);
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < kK; ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    // n_seen: margin observations held before this tick (the bank's mx_.size() before transform)
    void score(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        bool scorable = n_seen >= 30;
        for (const auto& a : arms) scorable = scorable && a->ready();
        if (!scorable) return;
        for (std::size_t k = 0; k < kK; ++k) {
            const double par = copula::param_from_spearman(copula::Family::Gaussian, arms[k]->estimate());
            loss[k] = -copula::log_density(copula::Family::Gaussian, u, v, par);
        }
        double worst = 0;
        for (double l : loss) if (std::isfinite(l)) worst = std::max(worst, l);
        for (double& l : loss) if (!std::isfinite(l)) l = worst;
        double top = -INFINITY;
        for (std::size_t k = 0; k < kK; ++k) {
            lw[k] = (w[k] > 0 ? std::log(w[k]) : -INFINITY) - eta * loss[k];
            top = std::max(top, lw[k]);
        }
        double tot = 0;
        for (std::size_t k = 0; k < kK; ++k) { w[k] = std::exp(lw[k] - top); tot += w[k]; }
        for (double& x : w) x = (1 - delta) * x / tot + delta / kK;
    }
};

}  // namespace thesis
