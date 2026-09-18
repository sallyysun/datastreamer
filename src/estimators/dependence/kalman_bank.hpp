#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "estimator.hpp"

// A posterior over the forgetting rate, instead of a guess at it.
//
// Section 14.8b measured the problem: sigma_omega2 moves RMSE about 4x more than
// any detector, and its best value differs per scenario across two and a half
// orders of magnitude. No single value is close to good everywhere, and the
// value every detector in this document wraps (1e-2) is right for two of five
// scenarios.
//
// So do not choose one. Run a bank of local-level filters at different
// sigma_omega2, score each by the predictive likelihood it assigns to the block
// actually observed, and report the weighted mean. That is Bayesian model
// averaging over the memory length, and it is structurally the same device as
// BOCPD -- which averages over run length and is the one clear structural win in
// this document (finding 2).
//
// The weights need their own dynamics or they converge once and never move
// again, which is the cumulative-estimator trap in a new place. A small
// switch_prob puts a Markov chain on the model index, exactly as BOCPD's hazard
// does on the run length.
//
// Cost: O(K) scalar updates per block on ONE shared block Spearman, so ~500
// bytes and no extra rank sorting. See section 14.16.
class KalmanBank : public DependenceEstimator {
public:
    struct Config {
        std::size_t block = 20;
        // Spans the range section 14.8b found the per-scenario optimum over.
        std::vector<double> sigma_omega2 = {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0};
        double switch_prob = 0.02;   // per-block probability the memory changes
        std::size_t burn_in_blocks = 2;
    };

    KalmanBank(Config config, const char* name)
        : config_(config), name_(name),
          z_(config.sigma_omega2.size(), 0.0),
          P_(config.sigma_omega2.size(), kPriorVariance),
          w_(config.sigma_omega2.size(),
             1.0 / static_cast<double>(config.sigma_omega2.size())) {
        xs_.reserve(config_.block);
        ys_.reserve(config_.block);
    }

    void update(double x, double y) override {
        xs_.push_back(x);
        ys_.push_back(y);
        if (xs_.size() < config_.block) return;

        const double y_obs = std::atanh(std::clamp(block_spearman(), -0.999, 0.999));
        const double m = static_cast<double>(config_.block);
        const double sigma_v2 = 1.06 / std::max(m - 3.0, 1.0);
        const std::size_t K = z_.size();

        // Score each model by the density it gave the observation BEFORE using it.
        std::vector<double> logl(K);
        for (std::size_t k = 0; k < K; ++k) {
            const double P_pred = P_[k] + config_.sigma_omega2[k];
            const double F = P_pred + sigma_v2;
            const double e = y_obs - z_[k];
            logl[k] = -0.5 * (std::log(F) + e * e / F);

            const double gain = P_pred / F;
            z_[k] += gain * e;
            P_[k] = (1.0 - gain) * P_pred;
        }

        // Bayes on the weights, in log space so a confident model cannot underflow.
        const double top = *std::max_element(logl.begin(), logl.end());
        double total = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            w_[k] *= std::exp(logl[k] - top);
            total += w_[k];
        }
        if (total > 0.0 && std::isfinite(total)) {
            for (double& wk : w_) wk /= total;
        } else {
            std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(K));
        }

        // Markov transition on the model index: the right memory length is itself
        // allowed to change, so the posterior stays able to move.
        const double floor = config_.switch_prob / static_cast<double>(K);
        for (double& wk : w_) wk = (1.0 - config_.switch_prob) * wk + floor;

        ++blocks_;
        xs_.clear();
        ys_.clear();
    }

    double estimate() const override {
        if (blocks_ < config_.burn_in_blocks) return 0.0;
        double z = 0.0;
        for (std::size_t k = 0; k < z_.size(); ++k) z += w_[k] * z_[k];
        return std::clamp(std::tanh(z), -1.0, 1.0);
    }

    void reset() override {
        xs_.clear();
        ys_.clear();
        std::fill(z_.begin(), z_.end(), 0.0);
        std::fill(P_.begin(), P_.end(), kPriorVariance);
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(z_.size()));
        blocks_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this)
             + (z_.capacity() + P_.capacity() + w_.capacity()
                + config_.sigma_omega2.capacity() + 2 * config_.block) * sizeof(double);
    }

    const char* name() const override { return name_; }

    std::size_t size() const { return w_.size(); }
    double weight(std::size_t k) const { return w_[k]; }
    double sigma_omega2(std::size_t k) const { return config_.sigma_omega2[k]; }

    // The posterior-mean memory length, for diagnostics.
    double effective_sigma_omega2() const {
        double q = 0.0;
        for (std::size_t k = 0; k < w_.size(); ++k) q += w_[k] * config_.sigma_omega2[k];
        return q;
    }

private:
    static constexpr double kPriorVariance = 1.0;

    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&v](std::size_t a, std::size_t b) { return v[a] < v[b]; });
        std::vector<double> ranks(n);
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            while (j + 1 < n && v[order[j + 1]] == v[order[i]]) ++j;
            const double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
            for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = avg;
            i = j + 1;
        }
        return ranks;
    }

    double block_spearman() const {
        const std::size_t n = xs_.size();
        const std::vector<double> rx = average_ranks(xs_);
        const std::vector<double> ry = average_ranks(ys_);
        const double mean = static_cast<double>(n + 1) / 2.0;
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = rx[i] - mean, dy = ry[i] - mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        const double denom = std::sqrt(var_x * var_y);
        return (denom > 0.0) ? cov / denom : 0.0;
    }

    Config config_;
    const char* name_;
    std::vector<double> z_, P_, w_;
    std::vector<double> xs_, ys_;
    std::size_t blocks_ = 0;
};
