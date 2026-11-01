#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../copula_family.hpp"
#include "estimator.hpp"
#include "marginal_rank_pit.hpp"
#include "predictive_density.hpp"

// Online bank of forgetting factors: thesis Algorithm 1 (Chapter 5, section 5.3),
// one tick at a time. Shared prequential margins, prediction loss per component
// (squared against p_t, or copula log loss), Hedge then fixed share, every
// component updated on the shared pseudo-observations, output the weighted rho.
// Log loss needs a density. Loss::Log borrows the method-of-moments copula
// c_{g^{-1}(rho_k)} of `family` (thesis section 4.3.1), so it sees only rho; Loss::Own
// asks each component for its own predictive density (PredictiveDensity), falling back
// to the method-of-moments density for a component that has none.
// notes/studies/thesis_bank_own_density.md.
// Design and how it differs from EstimatorBank: notes/thesis/chapter6_plan.md.
class ForgettingBank : public DependenceEstimator {
public:
    enum class Loss { Squared, Log, Own };

    struct Config {
        Loss loss = Loss::Squared;
        double eta = 0.5;              // Hedge learning rate
        double delta = 1.0e-4;         // fixed-share fraction per tick
        std::size_t pit_window = 500;  // shared marginal estimate (sliding-window ranks)
        std::size_t warm_up = 30;      // margin observations before the first score
        copula::Family family = copula::Family::Gaussian;
        double nu = 4.0;
    };

    ForgettingBank(std::vector<std::unique_ptr<DependenceEstimator>> models, Config config,
                   const char* name)
        : config_(config), name_(name), models_(std::move(models)),
          w_(models_.size(), 1.0 / static_cast<double>(models_.size())),
          loss_(models_.size(), 0.0), lw_(models_.size(), 0.0),
          pit_x_(config.pit_window), pit_y_(config.pit_window) {
        for (const auto& m : models_) dens_.push_back(dynamic_cast<const PredictiveDensity*>(m.get()));
    }

    void update(double x, double y) override {
        const std::size_t K = models_.size();
        const bool scorable = pit_x_.size() >= config_.warm_up;

        // Lines 5 and 13: transform with the margins held through t-1, then absorb.
        const double u = pit_x_.transform(x);
        const double v = pit_y_.transform(y);

        if (scorable) {
            for (std::size_t k = 0; k < K; ++k) loss_[k] = loss(k, u, v);
            // Guard: a non-finite loss is replaced by the worst finite loss this tick.
            double worst = 0.0;
            for (double l : loss_) if (std::isfinite(l)) worst = std::max(worst, l);
            for (double& l : loss_) if (!std::isfinite(l)) l = worst;

            // Hedge in log space, max subtracted before exponentiating.
            double top = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < K; ++k) {
                lw_[k] = (w_[k] > 0.0 ? std::log(w_[k]) : -std::numeric_limits<double>::infinity())
                       - config_.eta * loss_[k];
                top = std::max(top, lw_[k]);
            }
            double total = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                w_[k] = std::exp(lw_[k] - top);
                total += w_[k];
            }
            const double share = config_.delta / static_cast<double>(K);
            for (double& wk : w_) wk = (1.0 - config_.delta) * wk / total + share;
        }

        // Line 12: each component sees the shared pseudo-observations, as normal
        // scores so that real-line estimators (Hermite, the MLE) can take them.
        const double zx = RankPit::probit(u);
        const double zy = RankPit::probit(v);
        for (auto& m : models_) m->update(zx, zy);
        ++n_;
    }

    double estimate() const override {
        double r = 0.0;
        for (std::size_t k = 0; k < models_.size(); ++k) r += w_[k] * models_[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }

    // Predictive log density of the bank. Loss::Own: the weighted mixture of the components'
    // own densities (what the section 5.3.4 log-loss bound covers). Otherwise the
    // method-of-moments density at the bank's own rho, c_{g^{-1}(rho_bank)}(u, v).
    double log_density(double u, double v) const {
        if (config_.loss == Loss::Own) {
            double top = -std::numeric_limits<double>::infinity();
            std::vector<double> l(models_.size());
            for (std::size_t k = 0; k < models_.size(); ++k) {
                l[k] = std::log(w_[k]) - loss(k, u, v);
                top = std::max(top, l[k]);
            }
            if (!std::isfinite(top)) return top;
            double s = 0.0;
            for (double x : l) s += std::exp(x - top);
            return top + std::log(s);
        }
        return copula::log_density(config_.family, u, v,
                                   copula::param_from_spearman(config_.family, estimate(), config_.nu),
                                   config_.nu);
    }

    void reset() override {
        for (auto& m : models_) m->reset();
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(models_.size()));
        std::fill(loss_.begin(), loss_.end(), 0.0);
        pit_x_.reset();
        pit_y_.reset();
        n_ = 0;
    }

    std::size_t store_bytes() const override {
        std::size_t total = sizeof(*this) + pit_x_.store_bytes() + pit_y_.store_bytes()
                          + (w_.capacity() + loss_.capacity() + lw_.capacity()) * sizeof(double);
        for (const auto& m : models_) total += m->store_bytes();
        return total;
    }

    const char* name() const override { return name_; }

    std::size_t size() const { return models_.size(); }
    double weight(std::size_t k) const { return w_[k]; }
    double last_loss(std::size_t k) const { return loss_[k]; }
    const DependenceEstimator& model(std::size_t k) const { return *models_[k]; }

private:
    double loss(std::size_t k, double u, double v) const {
        const double rho_k = models_[k]->estimate();
        if (config_.loss == Loss::Own && dens_[k]) {
            const double l = dens_[k]->log_predictive(u, v);
            if (!std::isnan(l)) return -l;
        }
        if (config_.loss == Loss::Squared) {
            const double p = 12.0 * (u - 0.5) * (v - 0.5);
            return (p - rho_k) * (p - rho_k);
        }
        const double par = copula::param_from_spearman(config_.family, rho_k, config_.nu);
        return -copula::log_density(config_.family, u, v, par, config_.nu);
    }

    Config config_;
    const char* name_;
    std::vector<std::unique_ptr<DependenceEstimator>> models_;
    std::vector<double> w_, loss_, lw_;
    std::vector<const PredictiveDensity*> dens_;
    RankPit pit_x_, pit_y_;
    std::size_t n_ = 0;
};
