#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "estimator.hpp"
#include "marginal_rank_pit.hpp"

// KalmanBank's model averaging over memory for any estimator: model k is scored by the Gaussian
// predictive log L_k = -s_k^2/(2v), s_k = 12(u-1/2)(v-1/2) - rho_hat_k, per block, then Bayes.
// See notes/studies/adaptive_memory.md and notes/design/source_rationale.md.
class EstimatorBank : public DependenceEstimator {
public:
    struct Config {
        std::size_t block = 20;
        double switch_prob = 0.002;   // Markov step on the model index
        double var_rate = 0.001;      // must be far slower than the change (§14.11)
        std::size_t pit_window = 500;
        std::size_t burn_in_ticks = 200;
    };

    EstimatorBank(std::vector<std::unique_ptr<DependenceEstimator>> models,
                   Config config, const char* name)
        : config_(config), name_(name), models_(std::move(models)),
          w_(models_.size(), 1.0 / static_cast<double>(models_.size())),
          ll_(models_.size(), 0.0),
          pit_x_(config.pit_window), pit_y_(config.pit_window) {}

    void update(double x, double y) override {
        const std::size_t K = models_.size();

        // Score BEFORE absorbing, so each model is judged on a prediction.
        const double u = pit_x_.transform(x);
        const double v = pit_y_.transform(y);
        const bool scorable = (pit_x_.size() >= 30);
        const double point = 12.0 * (u - 0.5) * (v - 0.5);

        if (scorable) {
            double s_mix = point - estimate_raw();
            var_ += config_.var_rate * (s_mix * s_mix - var_);
            const double denom = 2.0 * std::max(var_, 1e-6);
            for (std::size_t k = 0; k < K; ++k) {
                const double s = point - models_[k]->estimate();
                ll_[k] -= s * s / denom;
            }
        }

        for (auto& m : models_) m->update(x, y);
        ++n_;

        if (!scorable || ++in_block_ < config_.block) return;
        in_block_ = 0;

        const double top = *std::max_element(ll_.begin(), ll_.end());
        double total = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            w_[k] *= std::exp(ll_[k] - top);
            ll_[k] = 0.0;
            total += w_[k];
        }
        if (total > 0.0 && std::isfinite(total)) {
            for (double& wk : w_) wk /= total;
        } else {
            std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(K));
        }
        const double floor = config_.switch_prob / static_cast<double>(K);
        for (double& wk : w_) wk = (1.0 - config_.switch_prob) * wk + floor;
    }

    double estimate() const override {
        if (n_ < config_.burn_in_ticks) return 0.0;
        return estimate_raw();
    }

    void reset() override {
        for (auto& m : models_) m->reset();
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(models_.size()));
        std::fill(ll_.begin(), ll_.end(), 0.0);
        pit_x_.reset();
        pit_y_.reset();
        var_ = 1.0;
        in_block_ = 0;
        n_ = 0;
    }

    std::size_t store_bytes() const override {
        std::size_t total = sizeof(*this) + pit_x_.store_bytes() + pit_y_.store_bytes()
                          + (w_.capacity() + ll_.capacity()) * sizeof(double);
        for (const auto& m : models_) total += m->store_bytes();
        return total;
    }

    const char* name() const override { return name_; }

    double weight(std::size_t k) const { return w_[k]; }
    std::size_t size() const { return models_.size(); }

private:
    double estimate_raw() const {
        double r = 0.0;
        for (std::size_t k = 0; k < models_.size(); ++k) r += w_[k] * models_[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }

    Config config_;
    const char* name_;
    std::vector<std::unique_ptr<DependenceEstimator>> models_;
    std::vector<double> w_, ll_;
    RankPit pit_x_, pit_y_;
    double var_ = 1.0;
    std::size_t in_block_ = 0, n_ = 0;
};
