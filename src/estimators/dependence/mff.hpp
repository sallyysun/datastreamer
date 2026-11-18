#pragma once

// Multiple forgetting factors (Chapter 5): Hedge with fixed share over fixed-memory arms, given the shared
// pseudo-observations (U, V) from outside.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "study_config.hpp"
#include "../../copula_family.hpp"
#include "estimator.hpp"

namespace thesis {

inline std::size_t guard_mff_losses(std::vector<double>& losses) {
    double worst = 0.0;
    std::size_t nonfinite = 0;
    for (double loss : losses) {
        if (std::isfinite(loss)) worst = std::max(worst, loss);
        else ++nonfinite;
    }
    for (double& loss : losses) {
        if (!std::isfinite(loss)) loss = worst;
    }
    return nonfinite;
}

// Apply Hedge in log space, then mix in uniform weight.
inline void apply_fixed_share(std::vector<double>& weights, const std::vector<double>& losses,
                              std::vector<double>& log_weights, double eta, double delta) {
    double top = -INFINITY;
    for (std::size_t k = 0; k < weights.size(); ++k) {
        const double log_weight = weights[k] > 0 ? std::log(weights[k]) : -INFINITY;
        log_weights[k] = log_weight - eta * losses[k];
        top = std::max(top, log_weights[k]);
    }
    double total = 0.0;
    for (std::size_t k = 0; k < weights.size(); ++k) {
        weights[k] = std::exp(log_weights[k] - top);
        total += weights[k];
    }
    const double uniform_share = delta / static_cast<double>(weights.size());
    for (double& weight : weights) weight = (1 - delta) * weight / total + uniform_share;
}

inline void update_mff_weights(std::vector<double>& weights, std::vector<double>& losses,
                               std::vector<double>& log_weights, double eta, double delta) {
    guard_mff_losses(losses);
    apply_fixed_share(weights, losses, log_weights, eta, delta);
}

// MFF over the study's fixed memories, scored under one working copula.
struct Mff {
    copula::Family family = copula::Family::Gaussian;
    double eta = 1.0, delta = 1e-4;
    std::vector<double> w = std::vector<double>(kK, 1.0 / kK), loss = std::vector<double>(kK), lw = std::vector<double>(kK);
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < kK; ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    // n_seen counts prior observations.
    bool ready(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, std::size_t n_seen) const {
        return n_seen >= 30 && std::all_of(arms.begin(), arms.end(), [](const auto& arm) { return arm->ready(); });
    }
    void score(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        if (!ready(arms, n_seen)) return;
        for (std::size_t k = 0; k < kK; ++k) {
            const double par = copula::param_from_spearman(family, arms[k]->estimate());
            loss[k] = -copula::log_density(family, u, v, par);
        }
        update_mff_weights(w, loss, lw, eta, delta);
    }
};

}  // namespace thesis
