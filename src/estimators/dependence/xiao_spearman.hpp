#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "estimator.hpp"

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
class XiaoSpearmanEstimator : public DependenceEstimator {
public:
    struct Config {
        std::size_t bins = 30;
        std::size_t window = 0;       // >0: hard sliding window
        double lambda = 0.0;          // in (0,1): retention factor; overrides window
        std::size_t calibration = 300;
    };

    XiaoSpearmanEstimator(Config config, const char* name)
        : config_(config),
          name_(name),
          counts_(config.bins * config.bins, 0.0),
          row_counts_(config.bins, 0.0),
          col_counts_(config.bins, 0.0) {
        calibration_.reserve(config.calibration);
    }

    void update(double x, double y) override {
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
        calibration_.reserve(config_.calibration);
    }

    std::size_t store_bytes() const override {
        return (counts_.size() + row_counts_.size() + col_counts_.size()) * sizeof(double)
             + (cuts_x_.size() + cuts_y_.size()) * sizeof(double)
             + window_.size() * sizeof(BinPair)
             + calibration_.capacity() * sizeof(std::pair<double, double>);
    }

    const char* name() const override { return name_; }

private:
    struct BinPair {
        std::uint16_t i;
        std::uint16_t j;
    };

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

    std::size_t bin_of(const std::vector<double>& cuts, double value) const {
        return static_cast<std::size_t>(
            std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
    }

    void insert(double x, double y) {
        const std::size_t i = bin_of(cuts_x_, x);
        const std::size_t j = bin_of(cuts_y_, y);
        const std::size_t m = config_.bins;

        if (config_.lambda > 0.0) {
            const double l = config_.lambda;
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
};
