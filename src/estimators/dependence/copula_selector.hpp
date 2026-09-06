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

// Online parametric-vs-nonparametric copula selection (the "GoF selection" item).
//
// Runs a bank of candidate copula models on shared pseudo-observations and scores
// each by its forgetting-weighted PREQUENTIAL log-likelihood: at every step a
// candidate first PREDICTS the incoming (u,v) with the model fitted on all prior
// points, we accumulate that predictive log-density, THEN the candidate updates.
//   S_m(t) = lambda * S_m(t-1) + log c_m( u_t, v_t ; params fitted on 1..t-1 )
// estimate() reports the Spearman rho of the currently best-scoring candidate.
//
// Why prequential rather than an in-sample AIC/BIC: the marginal transform is the
// SAME for every candidate (shared Welford standardization), so the marginal
// likelihood cancels and S_m is a like-for-like predictive score across copulas.
// One-step-ahead prediction also penalises the extra flexibility of the
// nonparametric member automatically (it predicts worse until it has seen enough),
// so no explicit complexity penalty is needed. This is the online analogue of a
// copula goodness-of-fit / model-selection test, and it makes the
// parametric-vs-nonparametric choice itself a streaming, O(1)/tick decision.
//
// The candidate set is: Gaussian, Clayton, Gumbel, Student-t (each a CopulaFit),
// plus a binned empirical-copula density (the nonparametric member). All four
// parametric fits share the CopulaFit code with ParametricCopulaMLE, so a family
// selected here behaves identically to running that family standalone.

namespace copula {

// Binned empirical-copula density on (u,v) -- the nonparametric candidate. A
// K x K uniform grid of forgetting-weighted counts; Laplace-smoothed so the
// predictive density is always finite (an empty grid predicts the independence
// copula, density 1).
class NonparamCopula {
public:
    NonparamCopula(int bins, double lambda, double alpha = 1.0)
        : bins_(bins), lambda_(lambda), alpha_(alpha), counts_(bins * bins, 0.0) {}

    void reset() {
        std::fill(counts_.begin(), counts_.end(), 0.0);
        total_ = 0.0;
    }

    void update_uv(double u, double v) {
        const int i = cell(u), j = cell(v);
        if (lambda_ > 0.0) {
            for (auto& c : counts_) c *= lambda_;
            total_ *= lambda_;
            counts_[i * bins_ + j] += (1.0 - lambda_);
            total_ += (1.0 - lambda_);
        } else {
            counts_[i * bins_ + j] += 1.0;
            total_ += 1.0;
        }
    }

    // Predictive copula log-density: smoothed cell probability / cell area (1/K^2).
    double logpdf_uv(double u, double v) const {
        const int i = cell(u), j = cell(v);
        const double k2 = static_cast<double>(bins_) * bins_;
        const double prob = (counts_[i * bins_ + j] + alpha_) / (total_ + alpha_ * k2);
        return std::log(prob * k2);
    }

    // Spearman's rho from the grid: 12 E[U V] - 3 with cell-centre coordinates.
    double spearman() const {
        if (total_ <= 0.0) return 0.0;
        double e_uv = 0.0;
        for (int i = 0; i < bins_; ++i) {
            const double uc = (i + 0.5) / bins_;
            for (int j = 0; j < bins_; ++j) {
                const double vc = (j + 0.5) / bins_;
                e_uv += counts_[i * bins_ + j] * uc * vc;
            }
        }
        return std::clamp(12.0 * (e_uv / total_) - 3.0, -1.0, 1.0);
    }

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
        double lambda = 0.0;           // forgetting for BOTH the fits and the scores
        std::size_t burn_in = 100;     // let the scores settle before selecting
        double nu = 4.0;               // Student-t degrees of freedom
        int nonparam_bins = 10;        // grid resolution of the nonparametric member
        bool include_nonparam = true;  // include the nonparametric candidate
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

        // Prequential: score with the current fit, THEN update it.
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

    // Diagnostics: which candidate currently wins (not part of the interface).
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
             + fits_.capacity() * sizeof(copula::CopulaFit);
    }
    const char* name() const override { return name_; }

private:
    static constexpr double kEps = 1e-9;

    // Best-scoring candidate index; -1 means the nonparametric member wins.
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
