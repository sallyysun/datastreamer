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

inline constexpr std::size_t kMffWarmup = 30;

// Hedge with fixed share over any set of arms. The default is MFF over the study's fixed memories under one
// working copula.
struct Mff {
    copula::Family family;
    double eta, delta;
    std::vector<double> w, loss, lw;

    explicit Mff(copula::Family family = copula::Family::Gaussian, std::size_t arms = kK, double eta = 1.0,
                 double delta = 1e-4)
        : family(family), eta(eta), delta(delta), w(arms, 1.0 / double(arms)), loss(arms), lw(arms) {}

    std::size_t size() const { return w.size(); }

    double forecast(const double* rho) const {
        double r = 0;
        for (std::size_t k = 0; k < w.size(); ++k) r += w[k] * rho[k];
        return std::clamp(r, -1.0, 1.0);
    }
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < w.size(); ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    // log sum_k w_k c_k from the arms' log densities, over arms with positive weight.
    double mixture_log_density(const double* log_density) const {
        double top = -INFINITY;
        for (std::size_t k = 0; k < w.size(); ++k)
            if (w[k] > 0) top = std::max(top, std::log(w[k]) + log_density[k]);
        double m = 0;
        for (std::size_t k = 0; k < w.size(); ++k)
            if (w[k] > 0) m += std::exp(std::log(w[k]) + log_density[k] - top);
        return top + std::log(m);
    }
    // -log sum_k w_k exp(-loss_k), shifted by the smallest loss.
    double mixture_loss(const double* losses) const {
        double lo = INFINITY;
        for (std::size_t k = 0; k < w.size(); ++k) lo = std::min(lo, losses[k]);
        double mix = 0;
        for (std::size_t k = 0; k < w.size(); ++k) mix += w[k] * std::exp(-(losses[k] - lo));
        return lo - std::log(mix);
    }
    // n_seen counts prior observations.
    bool ready(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, std::size_t n_seen) const {
        return n_seen >= kMffWarmup && std::all_of(arms.begin(), arms.end(), [](const auto& arm) { return arm->ready(); });
    }
    void update(const std::vector<double>& losses) { apply_fixed_share(w, losses, lw, eta, delta); }
    void score_and_update_weights(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        if (!ready(arms, n_seen)) return;
        for (std::size_t k = 0; k < w.size(); ++k) {
            const double par = copula::param_from_spearman(family, arms[k]->estimate());
            loss[k] = -copula::log_density(family, u, v, par);
        }
        guard_mff_losses(loss);
        update(loss);
    }
};

}  // namespace thesis
