#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "../copula_family.hpp"
#include "../special_functions.hpp"
#include "estimator.hpp"
#include "parametric_copula.hpp"

// Online parametric-vs-nonparametric copula selection by forgetting-weighted
// prequential log-likelihood. Rationale: notes/studies/roadmap.md.
namespace copula {

class NonparamCopula {
public:
    NonparamCopula(int bins, double lambda, double alpha = 1.0)
        : bins_(bins), lambda_(lambda), alpha_(alpha), counts_(bins * bins, 0.0) {}

    void reset() {
        std::fill(counts_.begin(), counts_.end(), 0.0);
        total_ = 0.0;
    }

    // Counts stay in observation units (total_ -> 1/(1-lambda)) so the alpha
    // prior is weighed against the effective sample size, not against 1.
    void update_uv(double u, double v) {
        const int i = cell(u), j = cell(v);
        if (lambda_ > 0.0) {
            for (auto& c : counts_) c *= lambda_;
            total_ *= lambda_;
        }
        counts_[i * bins_ + j] += 1.0;
        total_ += 1.0;
    }

    double logpdf_uv(double u, double v) const {
        const int i = cell(u), j = cell(v);
        const double k2 = static_cast<double>(bins_) * bins_;
        const double prob = (counts_[i * bins_ + j] + alpha_) / (total_ + alpha_ * k2);
        return std::log(prob * k2);
    }

    // Weighted Pearson of marginal mid-ranks, as XiaoSpearmanEstimator: the
    // weighted margins are not exactly uniform, so an uncentred 12E[uv]-3 picks
    // up their mean fluctuations.
    double spearman() const {
        if (total_ <= 0.0) return 0.0;
        std::vector<double> rx(bins_, 0.0), ry(bins_, 0.0);
        for (int i = 0; i < bins_; ++i) {
            for (int j = 0; j < bins_; ++j) {
                rx[i] += counts_[i * bins_ + j];
                ry[j] += counts_[i * bins_ + j];
            }
        }
        double var_x = 0.0, var_y = 0.0, cum_x = 0.0, cum_y = 0.0;
        for (int i = 0; i < bins_; ++i) {
            const double mx = rx[i], my = ry[i];
            rx[i] = cum_x + mx / 2.0 - total_ / 2.0;
            ry[i] = cum_y + my / 2.0 - total_ / 2.0;
            cum_x += mx;
            cum_y += my;
            var_x += mx * rx[i] * rx[i];
            var_y += my * ry[i] * ry[i];
        }
        const double denom = std::sqrt(var_x * var_y);
        if (denom <= 0.0) return 0.0;
        double cov = 0.0;
        for (int i = 0; i < bins_; ++i)
            for (int j = 0; j < bins_; ++j) cov += counts_[i * bins_ + j] * rx[i] * ry[j];
        return std::clamp(cov / denom, -1.0, 1.0);
    }

    std::size_t heap_bytes() const { return counts_.capacity() * sizeof(double); }

private:
    int cell(double u) const {
        int c = static_cast<int>(u * bins_);
        return std::clamp(c, 0, bins_ - 1);
    }

    int bins_;
    double lambda_;
    double alpha_;
    std::vector<double> counts_;
    double total_ = 0.0;
};

}  // namespace copula

class CopulaSelector : public DependenceEstimator {
public:
    struct Config {
        double lambda = 0.0;
        std::size_t burn_in = 100;
        double nu = 4.0;
        int nonparam_bins = 10;
        bool include_nonparam = true;
    };

    CopulaSelector(Config config, const char* name)
        : config_(config), name_(name), nonparam_(config.nonparam_bins, config.lambda) {
        using copula::Family;
        for (Family f : {Family::Gaussian, Family::Clayton, Family::Gumbel, Family::StudentT}) {
            fits_.emplace_back(f, config.nu, config.lambda);
            score_.push_back(0.0);
        }
    }

    void update(double x, double y) override {
        ++n_;
        update_marginals(x, y);
        if (n_ < 2) return;
        const double u = pseudo(x, mean_x_, var_x_);
        const double v = pseudo(y, mean_y_, var_y_);
        const double decay = (config_.lambda > 0.0) ? config_.lambda : 1.0;

        for (std::size_t m = 0; m < fits_.size(); ++m) {
            score_[m] = decay * score_[m] + finite(fits_[m].logpdf_uv(u, v));
            fits_[m].update_uv(u, v);
        }
        if (config_.include_nonparam) {
            score_np_ = decay * score_np_ + finite(nonparam_.logpdf_uv(u, v));
            nonparam_.update_uv(u, v);
        }
    }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0;
        const int best = best_index();
        if (best < 0) return nonparam_.spearman();
        return fits_[best].spearman();
    }

    std::string selected() const {
        const int best = best_index();
        if (best < 0) return "nonparam";
        switch (fits_[best].family()) {
            case copula::Family::Gaussian:  return "gaussian";
            case copula::Family::Clayton:   return "clayton";
            case copula::Family::Gumbel:    return "gumbel";
            case copula::Family::StudentT:  return "student_t";
        }
        return "?";
    }

    void reset() override {
        n_ = 0;
        for (auto& f : fits_) f.reset();
        std::fill(score_.begin(), score_.end(), 0.0);
        nonparam_.reset();
        score_np_ = 0.0;
        mean_x_ = mean_y_ = 0.0;
        m2_x_ = m2_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + score_.capacity() * sizeof(double)
             + fits_.capacity() * sizeof(copula::CopulaFit) + nonparam_.heap_bytes();
    }
    const char* name() const override { return name_; }

private:
    static constexpr double kEps = 1e-9;

    int best_index() const {
        int best = -2;
        double best_score = -1e300;
        for (std::size_t m = 0; m < fits_.size(); ++m) {
            if (score_[m] > best_score) { best_score = score_[m]; best = static_cast<int>(m); }
        }
        if (config_.include_nonparam && score_np_ > best_score) best = -1;
        return best;
    }

    static double finite(double v) { return std::isfinite(v) ? v : -50.0; }

    static double pseudo(double value, double mean, double var) {
        const double z = (value - mean) / std::sqrt(std::max(var, 1e-12));
        return std::clamp(special::normal_cdf(z), kEps, 1.0 - kEps);
    }

    void update_marginals(double x, double y) {
        const double n = static_cast<double>(n_);
        const double dx = x - mean_x_;
        const double dy = y - mean_y_;
        mean_x_ += dx / n;
        mean_y_ += dy / n;
        m2_x_ += dx * (x - mean_x_);
        m2_y_ += dy * (y - mean_y_);
        if (n_ > 1) {
            var_x_ = m2_x_ / (n - 1.0);
            var_y_ = m2_y_ / (n - 1.0);
        }
    }

    Config config_;
    const char* name_;
    std::vector<copula::CopulaFit> fits_;
    std::vector<double> score_;
    copula::NonparamCopula nonparam_;
    double score_np_ = 0.0;
    std::size_t n_ = 0;

    double mean_x_ = 0.0, mean_y_ = 0.0;
    double m2_x_ = 0.0, m2_y_ = 0.0;
    double var_x_ = 1.0, var_y_ = 1.0;
};
