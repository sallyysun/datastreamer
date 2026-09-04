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

// Bayesian Online Changepoint Detection (Adams & MacKay 2007) on blockwise
// Fisher-z'd Spearman's rho -- the Bayesian gold standard change_detection.md
// calls for (build order step 6), and the answer to the open question left by
// kalman_cusum.hpp / kalman_glr.hpp: does a SOFT run-length posterior avoid
// the hard-reset RMSE cost those two detect-and-restart methods paid on
// `jump`/`oscillate` (see notes/todo.md)?
//
// Model: y_t = atanh(windowed Spearman rho) every `block` ticks -- the same
// observation as KalmanLocalLevel -- with KNOWN sigma_v^2 ~ 1.06/(block-3).
// The per-segment mean mu=atanh(rho) is unknown with a conjugate Normal prior
// N(mu0, tau0^2). This is the roadmap's NIW-on-normal-scores construction
// SPECIALIZED to known observation variance: this whole Kalman/CUSUM/GLR line
// already treats sigma_v^2 as known (it has a closed form), so there is no
// reason to carry Normal-Inverse-Gamma machinery for a variance nobody needs
// estimated here.
//
// Run-length recursion (Adams & MacKay eq. 2-4), constant hazard H = 1/(mean
// segment length in blocks). r_{t-1}=r means "r observations belong to the
// current run as of the previous tick" (r=0 is the always-available fresh-
// prior hypothesis, sufficient stats identically empty). At each tick, EVERY
// hypothesis r contributes its own predictive likelihood L(y_t | r's stats)
// to BOTH possible next states: growth (r_t=r+1, weight 1-H) and reset
// (r_t=0, weight H, summed over every r -- a break could follow any prior
// hypothesis, each entering with evidence proportional to how well it
// explained the run so far). This is exact Bayesian model averaging: after
// normalizing, the r_t=0 bucket's PROBABILITY comes from a sum across every
// r_{t-1}, but its STATS reset to empty regardless of which r contributed the
// mass. Computed in log-space (log-sum-exp) throughout to avoid the
// underflow that many-observation predictive densities can produce.
//
// Truncated at `max_run_length` blocks (probability mass beyond it is
// dropped and the array renormalized) so this stays O(max_run_length) per
// block -- the same bounded-window trade already made for kalman_glr.hpp;
// an unbounded run-length array is not streaming-admissible.
//
// estimate() is the RUN-LENGTH-WEIGHTED POSTERIOR MEAN of mu (Bayesian model
// averaging over "how long has the current regime been running"), mapped
// through tanh. There is no hard reset anywhere in this class -- unlike
// kalman_cusum/kalman_glr, nothing here ever throws away the filter's state
// on an alarm; the "soft mixing" IS the whole mechanism.
class KalmanBocpdDetector : public DependenceEstimator, public ChangeDetector {
public:
    struct Config {
        std::size_t block = 20;
        double hazard_rate = 1.0 / 1000.0;  // H; matches the ARL0=1000-block target used to
                                             // calibrate kalman_cusum/kalman_glr, so all three
                                             // detectors share the same "expected time between
                                             // changes" operating point.
        double prior_mu0 = 0.0;             // prior mean of mu=atanh(rho) at a fresh run
        double prior_tau0_sq = 1.0;         // prior variance of mu at a fresh run
        std::size_t max_run_length = 150;   // truncation bound, in blocks (3000 ticks at block=20)
        // alarmed_this_block() fires when P(r_t <= short_run_cutoff | data)
        // exceeds this. NOT built on raw P(r_t=0): under a TRULY constant
        // hazard, P(r_t=0|data) is a fixed point of the recursion and equals
        // exactly `hazard_rate` on EVERY tick, independent of the data (the
        // one-step marginal likelihood common to the reset and growth
        // branches cancels algebraically in the normalization) -- verified
        // directly (flat at 0.0010 = 1/1000 across a real jump). The
        // detectable signal instead lives in how (1-H) redistributes across
        // the *growth* hypotheses r=1..short_run_cutoff, which genuinely
        // does concentrate near small r right after a real change (confirmed
        // via the MAP run length correctly collapsing to 1 at a synthetic
        // jump). short_run_cutoff=2 means "posterior belief the current run
        // is at most 2 blocks old."
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
        // Suppress alarms while the run-length array is itself still shorter
        // than short_run_cutoff+2: with too few blocks of history, EVERY
        // active hypothesis satisfies r<=short_run_cutoff trivially (there
        // has not been time for a longer-run alternative to even exist), so
        // P(short run) is deterministically ~1 regardless of data -- verified
        // directly: 100% spurious alarm at blocks 0-1 on a stationary stream
        // across 200 seeds, independent of sigma_omega2 or any real evidence.
        // This guard applies after EVERY reset() (including a genuine
        // detect-and-... well, there is no hard reset here, but external
        // reset() calls between harness runs hit it too), not just at
        // startup, which is exactly where it needs to apply.
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
