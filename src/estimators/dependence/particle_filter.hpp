#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "estimator.hpp"

// Bootstrap particle filter on Fisher-z'd blockwise Spearman's rho.
//
// The Kalman filter is optimal only if the state moves by Gaussian increments.
// That assumption is what forces a single memory length: one sigma_omega2 has to
// serve both quiet stretches and breaks. A particle filter carries the posterior
// as samples instead, so the increment distribution is free — here a jump-diffusion
// mixture, mostly small steps with a rare large one. That gives long memory while
// a regime holds and an instant move when it does not, without a detector and
// without a reset.
//
// O(P) per block rather than O(1). See section 0.3 and section 10 of
// notes/studies/kalman_changepoint_results.md.
class ParticleFilter : public DependenceEstimator {
public:
    struct Config {
        std::size_t block = 20;
        std::size_t particles = 400;
        double sigma_omega2 = 1e-4;    // diffusion component
        double jump_prob = 0.01;       // mixture weight on the jump component
        double jump_scale = 0.5;       // jump sd, in Fisher-z units
        double nu = 5.0;               // observation t degrees of freedom; large -> Gaussian
        std::size_t burn_in_blocks = 2;
        std::uint64_t seed = 20260908;
    };

    ParticleFilter(Config config, const char* name)
        : config_(config), name_(name), rng_(config.seed),
          z_(config.particles, 0.0), w_(config.particles, 1.0 / config.particles) {
        xs_.reserve(config_.block);
        ys_.reserve(config_.block);
    }

    void update(double x, double y) override {
        xs_.push_back(x);
        ys_.push_back(y);
        if (xs_.size() < config_.block) return;

        const double rho_s = block_spearman();
        const double y_obs = std::atanh(std::clamp(rho_s, -0.999, 0.999));
        const double m = static_cast<double>(config_.block);
        const double sigma_v = std::sqrt(1.06 / std::max(m - 3.0, 1.0));

        std::normal_distribution<double> diffusion(0.0, std::sqrt(config_.sigma_omega2));
        std::normal_distribution<double> jump(0.0, config_.jump_scale);
        std::uniform_real_distribution<double> unit(0.0, 1.0);

        double total = 0.0;
        for (std::size_t i = 0; i < z_.size(); ++i) {
            z_[i] += diffusion(rng_);
            if (unit(rng_) < config_.jump_prob) z_[i] += jump(rng_);

            // Student-t observation likelihood: a block Spearman is not Gaussian,
            // and a heavy tail stops one odd block from collapsing the weights.
            const double r = (y_obs - z_[i]) / sigma_v;
            w_[i] = std::pow(1.0 + r * r / config_.nu, -(config_.nu + 1.0) / 2.0);
            total += w_[i];
        }
        if (total <= 0.0 || !std::isfinite(total)) {
            std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(z_.size()));
        } else {
            for (double& wi : w_) wi /= total;
        }

        if (effective_sample_size() < 0.5 * static_cast<double>(z_.size())) resample();

        ++blocks_;
        xs_.clear();
        ys_.clear();
    }

    double estimate() const override {
        if (blocks_ < config_.burn_in_blocks) return 0.0;
        double mean = 0.0;
        for (std::size_t i = 0; i < z_.size(); ++i) mean += w_[i] * z_[i];
        return std::clamp(std::tanh(mean), -1.0, 1.0);
    }

    void reset() override {
        xs_.clear();
        ys_.clear();
        std::fill(z_.begin(), z_.end(), 0.0);
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(z_.size()));
        blocks_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + (z_.capacity() + w_.capacity()) * sizeof(double)
             + 2 * config_.block * sizeof(double);
    }

    const char* name() const override { return name_; }

private:
    double effective_sample_size() const {
        double s = 0.0;
        for (double wi : w_) s += wi * wi;
        return (s > 0.0) ? 1.0 / s : 0.0;
    }

    // Systematic resampling: one uniform draw, lower variance than multinomial.
    void resample() {
        const std::size_t P = z_.size();
        std::vector<double> next(P);
        std::uniform_real_distribution<double> unit(0.0, 1.0 / static_cast<double>(P));
        double u = unit(rng_), c = w_[0];
        std::size_t j = 0;
        for (std::size_t i = 0; i < P; ++i) {
            const double target = u + static_cast<double>(i) / static_cast<double>(P);
            while (target > c && j + 1 < P) c += w_[++j];
            next[i] = z_[j];
        }
        z_.swap(next);
        std::fill(w_.begin(), w_.end(), 1.0 / static_cast<double>(P));
    }

    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
        std::vector<double> r(n);
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) ++j;
            const double avg = static_cast<double>(i + j) / 2.0 + 1.0;
            for (std::size_t k = i; k <= j; ++k) r[idx[k]] = avg;
            i = j + 1;
        }
        return r;
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
    std::mt19937_64 rng_;
    std::vector<double> z_;
    std::vector<double> w_;
    std::vector<double> xs_;
    std::vector<double> ys_;
    std::size_t blocks_ = 0;
};
