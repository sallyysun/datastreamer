#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "estimator.hpp"
#include "marginal_rank_pit.hpp"

// Streaming empirical-copula estimator of Spearman's rho (the "Xiao" branch).
//
// C++ port of Xiao (2019), "Novel Online Algorithms for Nonparametric
// Correlations with Application to Analyze Sensor Data" (IEEE Big Data),
// reference implementation: external/onlineNPCORR (corr.py online_corr /
// utility.py spearmanr_by_matrix, get_index). This estimate() reproduces
// spearmanr_by_matrix exactly; we add two extensions beyond the paper:
// auto-calibrated cutpoints (below) and exponential forgetting (lambda).
//
// Marginals are discretised into `bins` equiprobable cells whose cutpoints are
// fixed from an initial calibration window. A bins x bins count grid then
// approximates the joint distribution, and Spearman's rho is read off as a
// rank correlation over the grid.
//
// Three memory regimes share the same estimate() formula (it is scale
// invariant, so the total weight need not be an integer):
//   * lambda > 0  : exponential forgetting  (counts *= lambda each step)
//   * window > 0  : hard sliding window      (oldest cell decremented)
//   * otherwise   : cumulative               (all history)
//
// rank_pit_window > 0 replaces the calibrated cutpoints with rank-PIT: each
// marginal goes through RankPit, so the cutpoints are exactly k/bins and no
// calibration window is needed. Section 14.19 of
// notes/studies/kalman_changepoint_results.md.
class XiaoSpearmanEstimator : public DependenceEstimator {
public:
    struct Config {
        std::size_t bins = 30;
        std::size_t window = 0;       // >0: hard sliding window
        double lambda = 0.0;          // in (0,1): retention factor; overrides window
        std::size_t calibration = 300;
        std::size_t rank_pit_window = 0;  // >0: bin on rank-PIT pseudo-observations

        // Variable forgetting, driven by the section 14.14 rank residual
        //   s = 12 (u - 1/2)(v - 1/2) - rho_hat,
        // over its KNOWN null variance (see vff_step). C_ then tracks E[e~^2],
        // which is 1 when the memory is right -- the same statistic and the same
        // multiplicative rule as KalmanLocalLevel's vff_rate. Mirror image: here
        // it drives alpha = 1 - lambda, so C_ > 1 LOWERS lambda.
        // See notes/studies/adaptive_memory.md.
        double vff_rate = 0.0;             // 0 disables
        std::size_t vff_block = 20;        // ticks per C_ update, as the Kalman's
        double vff_null_var = 1.0;         // section 14.14 measured 0.92-1.00
        std::size_t vff_pit_window = 500;
        double vff_mem_min = 25.0;         // clamps, in ticks of effective memory
        double vff_mem_max = 1000.0;
    };

    XiaoSpearmanEstimator(Config config, const char* name)
        : config_(config),
          name_(name),
          counts_(config.bins * config.bins, 0.0),
          row_counts_(config.bins, 0.0),
          col_counts_(config.bins, 0.0),
          pit_x_(config.rank_pit_window),
          pit_y_(config.rank_pit_window) {
        if (config_.rank_pit_window > 0) {
            use_uniform_cutpoints();
        } else {
            calibration_.reserve(config_.calibration);
        }
    }

    void update(double x, double y) override {
        if (config_.vff_rate > 0.0) vff_step(x, y);
        if (config_.rank_pit_window > 0) {
            insert(pit_x_.transform(x), pit_y_.transform(y));
            return;
        }
        if (!calibrated_) {
            calibration_.emplace_back(x, y);
            if (calibration_.size() >= config_.calibration) {
                build_cutpoints();
                for (const auto& p : calibration_) insert(p.first, p.second);
                calibration_.clear();
                calibration_.shrink_to_fit();
                calibrated_ = true;
            }
            return;
        }
        insert(x, y);
    }

    double estimate() const override {
        if (!calibrated_ || total_ <= 0.0) return 0.0;
        const std::size_t m = config_.bins;

        // Mid-rank of each marginal cell, centred on the grand mean (total_/2).
        std::vector<double> rx(m), ry(m);
        double cum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            rx[i] = cum + row_counts_[i] / 2.0 - total_ / 2.0;
            cum += row_counts_[i];
        }
        cum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            ry[j] = cum + col_counts_[j] / 2.0 - total_ / 2.0;
            cum += col_counts_[j];
        }

        double var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < m; ++i) var_x += row_counts_[i] * rx[i] * rx[i];
        for (std::size_t j = 0; j < m; ++j) var_y += col_counts_[j] * ry[j] * ry[j];

        const double denom = std::sqrt(var_x * var_y);
        if (denom <= 0.0) return 0.0;

        double cov = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                const double c = counts_[i * m + j];
                if (c != 0.0) cov += c * rx[i] * ry[j];
            }
        }
        return std::clamp(cov / denom, -1.0, 1.0);
    }

    void reset() override {
        std::fill(counts_.begin(), counts_.end(), 0.0);
        std::fill(row_counts_.begin(), row_counts_.end(), 0.0);
        std::fill(col_counts_.begin(), col_counts_.end(), 0.0);
        window_.clear();
        total_ = 0.0;
        calibrated_ = false;
        calibration_.clear();
        lambda_ = config_.lambda;
        vff_pit_x_.reset();
        vff_pit_y_.reset();
        vff_c_ = 1.0;
        vff_sum_ = 0.0;
        vff_in_block_ = 0;
        vff_blocks_ = 0;
        if (config_.rank_pit_window > 0) {
            pit_x_.reset();
            pit_y_.reset();
            use_uniform_cutpoints();
        } else {
            calibration_.reserve(config_.calibration);
        }
    }

    std::size_t store_bytes() const override {
        return (counts_.size() + row_counts_.size() + col_counts_.size()) * sizeof(double)
             + (cuts_x_.size() + cuts_y_.size()) * sizeof(double)
             + window_.size() * sizeof(BinPair)
             + calibration_.capacity() * sizeof(std::pair<double, double>)
             + (config_.rank_pit_window > 0 ? pit_x_.store_bytes() + pit_y_.store_bytes() : 0)
             + (config_.vff_rate > 0.0 ? vff_pit_x_.store_bytes() + vff_pit_y_.store_bytes() : 0);
    }

    const char* name() const override { return name_; }

    double effective_memory_ticks() const {
        const double alpha = 1.0 - lambda_;
        return (alpha > 0.0) ? (2.0 - alpha) / alpha : 0.0;
    }

private:
    struct BinPair {
        std::uint16_t i;
        std::uint16_t j;
    };

    void use_uniform_cutpoints() {
        cuts_x_.resize(config_.bins - 1);
        for (std::size_t k = 1; k < config_.bins; ++k) {
            cuts_x_[k - 1] = static_cast<double>(k) / static_cast<double>(config_.bins);
        }
        cuts_y_ = cuts_x_;
        calibrated_ = true;
    }

    void build_cutpoints() {
        std::vector<double> xs, ys;
        xs.reserve(calibration_.size());
        ys.reserve(calibration_.size());
        for (const auto& p : calibration_) {
            xs.push_back(p.first);
            ys.push_back(p.second);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        cuts_x_.resize(config_.bins - 1);
        cuts_y_.resize(config_.bins - 1);
        for (std::size_t k = 1; k < config_.bins; ++k) {
            const std::size_t idx = k * xs.size() / config_.bins;
            cuts_x_[k - 1] = xs[std::min(idx, xs.size() - 1)];
            cuts_y_[k - 1] = ys[std::min(idx, ys.size() - 1)];
        }
    }

    void vff_step(double x, double y) {
        // Residual against the estimate held BEFORE this observation is absorbed.
        const double rho_hat = estimate();
        const double u = vff_pit_x_.transform(x);
        const double v = vff_pit_y_.transform(y);
        if (vff_pit_x_.size() < 30) return;
        const double p = 12.0 * (u - 0.5) * (v - 0.5);
        const double s = p - rho_hat;

        // Fixed null scale, not an estimated one. An EWMA of s^2 chases the very
        // lag the VFF exists to remove (14.11's trap); E[p^2] - rho_hat^2 biases
        // the other way, since rho_hat is the lagging quantity. Section 14.14
        // measured Var(s) at 0.92-1.00 across the whole rho range, so 1 is a
        // KNOWN scale to within 8% and nothing has to be estimated. This is the
        // nonparametric stand-in for the Kalman's F.
        vff_sum_ += s * s / config_.vff_null_var;
        if (++vff_in_block_ < config_.vff_block) return;
        const double e2 = vff_sum_ / static_cast<double>(config_.vff_block);
        vff_sum_ = 0.0;
        vff_in_block_ = 0;
        if (++vff_blocks_ <= kVffBurnInBlocks) return;

        vff_c_ += config_.vff_rate * (e2 - vff_c_);
        double alpha = (1.0 - lambda_) * std::pow(std::max(vff_c_, 1e-6), config_.vff_rate);
        alpha = std::clamp(alpha, 2.0 / (config_.vff_mem_max + 1.0),
                                  2.0 / (config_.vff_mem_min + 1.0));
        lambda_ = 1.0 - alpha;
    }

    std::size_t bin_of(const std::vector<double>& cuts, double value) const {
        return static_cast<std::size_t>(
            std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
    }

    void insert(double x, double y) {
        const std::size_t i = bin_of(cuts_x_, x);
        const std::size_t j = bin_of(cuts_y_, y);
        const std::size_t m = config_.bins;

        if (config_.lambda > 0.0) {
            const double l = lambda_;
            for (auto& c : counts_) c *= l;
            for (auto& c : row_counts_) c *= l;
            for (auto& c : col_counts_) c *= l;
            total_ *= l;
            counts_[i * m + j] += (1.0 - l);
            row_counts_[i] += (1.0 - l);
            col_counts_[j] += (1.0 - l);
            total_ += (1.0 - l);
            return;
        }

        counts_[i * m + j] += 1.0;
        row_counts_[i] += 1.0;
        col_counts_[j] += 1.0;
        total_ += 1.0;

        if (config_.window > 0) {
            window_.push_back({static_cast<std::uint16_t>(i), static_cast<std::uint16_t>(j)});
            if (window_.size() > config_.window) {
                const BinPair old = window_.front();
                window_.pop_front();
                counts_[old.i * m + old.j] -= 1.0;
                row_counts_[old.i] -= 1.0;
                col_counts_[old.j] -= 1.0;
                total_ -= 1.0;
            }
        }
    }

    Config config_;
    const char* name_;
    bool calibrated_ = false;
    double total_ = 0.0;

    std::vector<double> counts_;
    std::vector<double> row_counts_;
    std::vector<double> col_counts_;
    std::vector<double> cuts_x_;
    std::vector<double> cuts_y_;
    std::deque<BinPair> window_;
    std::vector<std::pair<double, double>> calibration_;

    // Declared AFTER config_: a RankPit placed before it reads an uninitialised
    // window size (the parametric_copula.hpp bug of 2026-09).
    RankPit pit_x_;
    RankPit pit_y_;
    RankPit vff_pit_x_{config_.vff_pit_window};
    RankPit vff_pit_y_{config_.vff_pit_window};

    static constexpr std::size_t kVffBurnInBlocks = 5;
    double lambda_ = config_.lambda;
    double vff_c_ = 1.0;
    double vff_sum_ = 0.0;
    std::size_t vff_in_block_ = 0;
    std::size_t vff_blocks_ = 0;
};
