#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <vector>

#include "change_detector.hpp"
#include "estimator.hpp"

// Bayesian Online Changepoint Detection (Adams & MacKay 2007) on the same
// blockwise Fisher-z'd Spearman observation as KalmanLocalLevel.
// Methodology and results: notes/studies/kalman_changepoint_results.md.
//
// The per-segment mean mu=atanh(rho) has a conjugate Normal prior N(mu0,
// tau0^2) with sigma_v^2 known -- the NIW construction specialized to known
// observation variance, since sigma_v^2 has a closed form here.
//
// Run-length recursion under a constant hazard H, where r_{t-1}=r means "r
// observations belong to the current run as of the previous tick" (r=0 is the
// fresh-prior hypothesis, stats empty). Each hypothesis contributes its own
// predictive likelihood to both successors: growth (r_t=r+1, weight 1-H) and
// reset (r_t=0, weight H, summed over all r). The r_t=0 bucket takes
// probability from every r but its stats reset to empty. Kept in log-space
// (log-sum-exp) to avoid underflow, and truncated at `max_run_length` so the
// cost stays O(max_run_length) per block.
//
// estimate() is the run-length-weighted posterior mean of mu mapped through
// tanh. Nothing here ever resets the state on an alarm -- the soft mixing is
// the mechanism.
class KalmanBocpdDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        std::size_t block = 20;
        double hazard_rate = 1.0 / 1000.0;  // H = 1 / mean segment length, in blocks
        double prior_mu0 = 0.0;             // prior mean of mu = atanh(rho) at a fresh run
        double prior_tau0_sq = 1.0;         // prior variance of mu at a fresh run
        std::size_t max_run_length = 150;   // truncation bound, in blocks
        // The alarm thresholds P(r_t <= short_run_cutoff | data), NOT raw
        // P(r_t=0 | data): under a constant hazard the latter equals
        // hazard_rate identically on every tick, independent of the data,
        // because the one-step marginal likelihood common to the reset and
        // growth branches cancels in the normalization. The usable signal is
        // how (1-H) redistributes across the growth hypotheses.
        std::size_t short_run_cutoff = 2;
        double alarm_prob_threshold = 0.5;
        std::size_t burn_in_blocks = 2;
    };

    explicit KalmanBocpdDetector(Config config, const char* name)
        : config_(config), name_(name) {
        xs_.reserve(config_.block);
        ys_.reserve(config_.block);
        reset();
    }

    void update(double x, double y) override {
        xs_.push_back(x);
        ys_.push_back(y);
        alarmed_ = false;
        if (xs_.size() < config_.block) return;

        const double rho_s = block_spearman();
        const double obs = std::atanh(std::clamp(rho_s, -0.999, 0.999));
        const double m = static_cast<double>(config_.block);
        const double sigma_v2 = 1.06 / std::max(m - 3.0, 1.0);
        xs_.clear();
        ys_.clear();
        ++blocks_;

        const double log_h = std::log(config_.hazard_rate);
        const double log_1mh = std::log(1.0 - config_.hazard_rate);
        const std::size_t R = log_prob_.size();  // active hypotheses, r=0..R-1

        std::vector<double> loglik(R), post_mu(R);
        for (std::size_t r = 0; r < R; ++r) {
            const double n = static_cast<double>(r);
            const double prec = 1.0 / config_.prior_tau0_sq + n / sigma_v2;
            const double tau_sq = 1.0 / prec;
            const double mu = tau_sq * (config_.prior_mu0 / config_.prior_tau0_sq + sum_y_[r] / sigma_v2);
            const double pred_var = tau_sq + sigma_v2;
            loglik[r] = -0.5 * std::log(2.0 * std::numbers::pi * pred_var) - 0.5 * (obs - mu) * (obs - mu) / pred_var;

            // Posterior mean AFTER incorporating obs (n+1 observations) --
            // used below for growth's stats and the model-averaged estimate.
            const double prec_post = 1.0 / config_.prior_tau0_sq + (n + 1.0) / sigma_v2;
            post_mu[r] = (1.0 / prec_post) * (config_.prior_mu0 / config_.prior_tau0_sq + (sum_y_[r] + obs) / sigma_v2);
        }

        // New array has support 0..R (R+1 entries): index 0 = reset, index
        // r+1 = growth from old index r.
        std::vector<double> new_log_prob(R + 1), new_sum_y(R + 1), new_post_mu(R + 1);
        double cp_logmass = -std::numeric_limits<double>::infinity();
        for (std::size_t r = 0; r < R; ++r) {
            cp_logmass = log_add_exp(cp_logmass, log_prob_[r] + loglik[r] + log_h);
        }
        new_log_prob[0] = cp_logmass;
        new_sum_y[0] = 0.0;
        // The reset target's posterior mean uses EMPTY prior stats plus this
        // one observation (n=1 total) -- exactly what post_mu[0] already is
        // (the r=0 entry above used sum_y_[0]=0, n=0 prior obs). R is always
        // >=1 here (log_prob_ starts at size 1 in reset()), so post_mu[0] is
        // always defined.
        new_post_mu[0] = post_mu[0];
        for (std::size_t r = 0; r < R; ++r) {
            new_log_prob[r + 1] = log_prob_[r] + loglik[r] + log_1mh;
            new_sum_y[r + 1] = sum_y_[r] + obs;
            new_post_mu[r + 1] = post_mu[r];
        }

        // Normalize in log-space, then truncate to max_run_length and
        // renormalize (dropping negligible tail mass -- the truncation this
        // class's O(max_run_length) bound requires).
        double log_norm = -std::numeric_limits<double>::infinity();
        for (double lp : new_log_prob) log_norm = log_add_exp(log_norm, lp);
        const std::size_t keep = std::min(new_log_prob.size(), config_.max_run_length + 1);
        log_prob_.assign(new_log_prob.begin(), new_log_prob.begin() + static_cast<long>(keep));
        sum_y_.assign(new_sum_y.begin(), new_sum_y.begin() + static_cast<long>(keep));
        std::vector<double> post_mu_kept(new_post_mu.begin(), new_post_mu.begin() + static_cast<long>(keep));

        double log_norm_kept = -std::numeric_limits<double>::infinity();
        for (double& lp : log_prob_) { lp -= log_norm; log_norm_kept = log_add_exp(log_norm_kept, lp); }
        for (double& lp : log_prob_) lp -= log_norm_kept;  // renormalize after truncation

        // Model-averaged point estimate and MAP-adjacent alarm signal.
        double mu_hat = 0.0;
        double best_lp = -std::numeric_limits<double>::infinity();
        for (std::size_t r = 0; r < log_prob_.size(); ++r) {
            const double p = std::exp(log_prob_[r]);
            mu_hat += p * post_mu_kept[r];
            best_lp = std::max(best_lp, log_prob_[r]);
        }
        estimate_ = std::clamp(std::tanh(mu_hat), -1.0, 1.0);
        p_changepoint_ = std::exp(log_prob_[0]);  // == hazard_rate identically; see the Config comment
        p_short_run_ = 0.0;
        for (std::size_t r = 0; r < log_prob_.size() && r <= config_.short_run_cutoff; ++r) {
            p_short_run_ += std::exp(log_prob_[r]);
        }
        // With fewer than short_run_cutoff+2 blocks of history every active
        // hypothesis satisfies r <= short_run_cutoff trivially (no longer-run
        // alternative has had time to exist), so P(short run) is ~1 regardless
        // of the data. Suppress alarms until there is enough history for the
        // comparison to mean anything.
        const bool long_enough_history = blocks_ > config_.short_run_cutoff + 1;
        alarmed_ = long_enough_history && (p_short_run_ > config_.alarm_prob_threshold);
        if (alarmed_) ++alarm_count_;
    }

    double estimate() const override {
        return blocks_ < config_.burn_in_blocks ? 0.0 : estimate_;
    }

    void reset() override {
        xs_.clear();
        ys_.clear();
        blocks_ = 0;
        log_prob_.assign(1, 0.0);  // P(r_0=0)=1, certainty at the very start
        sum_y_.assign(1, 0.0);
        estimate_ = 0.0;
        p_changepoint_ = 1.0;
        p_short_run_ = 1.0;
        alarmed_ = false;
        alarm_count_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.block * sizeof(double) +
               2 * (config_.max_run_length + 1) * sizeof(double);
    }

    const char* name() const override { return name_; }

    bool alarmed_this_block() const override { return alarmed_; }
    std::size_t alarm_count() const { return alarm_count_; }
    double p_changepoint() const { return p_changepoint_; }  // P(r_t=0 | data); == hazard_rate identically, see Config
    double p_short_run() const { return p_short_run_; }      // P(r_t <= short_run_cutoff | data); the real alarm signal
    std::size_t map_run_length() const {
        std::size_t best = 0;
        double best_lp = log_prob_.empty() ? 0.0 : log_prob_[0];
        for (std::size_t r = 1; r < log_prob_.size(); ++r) {
            if (log_prob_[r] > best_lp) { best_lp = log_prob_[r]; best = r; }
        }
        return best;
    }

private:
    static double log_add_exp(double a, double b) {
        if (a == -std::numeric_limits<double>::infinity()) return b;
        if (b == -std::numeric_limits<double>::infinity()) return a;
        const double m = std::max(a, b);
        return m + std::log(std::exp(a - m) + std::exp(b - m));
    }

    double block_spearman() const {
        const std::size_t n = xs_.size();
        const std::vector<double> rx = average_ranks(xs_);
        const std::vector<double> ry = average_ranks(ys_);

        const double mean = static_cast<double>(n + 1) / 2.0;
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = rx[i] - mean;
            const double dy = ry[i] - mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        const double denom = std::sqrt(var_x * var_y);
        return (denom > 0.0) ? cov / denom : 0.0;
    }

    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&v](std::size_t a, std::size_t b) -> bool { return v[a] < v[b]; });

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

    Config config_;
    const char* name_;

    std::vector<double> xs_, ys_;
    std::size_t blocks_ = 0;

    std::vector<double> log_prob_;  // log P(r_{t-1}=r | data), r=0..R-1
    std::vector<double> sum_y_;     // sufficient stat: sum of the r observations in hypothesis r's run

    double estimate_ = 0.0;
    double p_changepoint_ = 1.0;
    double p_short_run_ = 1.0;
    bool alarmed_ = false;
    std::size_t alarm_count_ = 0;
};
