#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "estimator.hpp"
#include "../special_functions.hpp"

// Counting Matrix estimator of Spearman's rho (thesis 4.1): C++ port of Xiao (2019), "Novel Online Algorithms
// for Nonparametric Correlations with Application to Analyze Sensor Data" (onlineNPCORR, spearmanr_by_matrix),
// with exponential forgetting.
//
// Each margin is cut into `bins` cells at cut-points fixed from the first `calibration` observations (or at
// probit(k/bins) when the inputs are already normal scores); a bins x bins count grid approximates the joint
// distribution and rho is the rank correlation over the grid. Forgetting: counts *= lambda, the new cell
// += 1 - lambda, with retention lambda = (N_eff - 1)/(N_eff + 1); lambda = 0 is cumulative.
class XiaoSpearmanEstimator : public DependenceEstimator {
public:
    struct Config {
        std::size_t bins = 30;
        double lambda = 0.0;
        std::size_t calibration = 300;
        bool normal_score_inputs = false;
    };

    XiaoSpearmanEstimator(Config config, const char* name)
        : config_(config), name_(name),
          counts_(config.bins * config.bins, 0.0), row_counts_(config.bins, 0.0), col_counts_(config.bins, 0.0) {
        start();
    }

    void update(double x, double y) override {
        if (calibrated_) { insert(x, y); return; }
        calibration_.emplace_back(x, y);
        if (calibration_.size() < config_.calibration) return;
        build_cutpoints();
        for (const auto& p : calibration_) insert(p.first, p.second);
        calibration_.clear();
        calibration_.shrink_to_fit();
        calibrated_ = true;
    }

    double estimate() const override {
        if (!calibrated_ || total_ <= 0.0) return 0.0;
        const std::size_t m = config_.bins;
        // Mid-rank of each marginal cell, centred on the grand mean total_/2.
        std::vector<double> rx(m), ry(m);
        double cum = 0.0;
        for (std::size_t i = 0; i < m; ++i) { rx[i] = cum + row_counts_[i] / 2.0 - total_ / 2.0; cum += row_counts_[i]; }
        cum = 0.0;
        for (std::size_t j = 0; j < m; ++j) { ry[j] = cum + col_counts_[j] / 2.0 - total_ / 2.0; cum += col_counts_[j]; }
        double var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < m; ++i) var_x += row_counts_[i] * rx[i] * rx[i];
        for (std::size_t j = 0; j < m; ++j) var_y += col_counts_[j] * ry[j] * ry[j];
        const double denom = std::sqrt(var_x * var_y);
        if (denom <= 0.0) return 0.0;
        double cov = 0.0;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const double c = counts_[i * m + j];
                if (c != 0.0) cov += c * rx[i] * ry[j];
            }
        return std::clamp(cov / denom, -1.0, 1.0);
    }

    void reset() override {
        std::fill(counts_.begin(), counts_.end(), 0.0);
        std::fill(row_counts_.begin(), row_counts_.end(), 0.0);
        std::fill(col_counts_.begin(), col_counts_.end(), 0.0);
        total_ = 0.0;
        calibrated_ = false;
        calibration_.clear();
        lambda_ = config_.lambda;
        start();
    }

    // The object plus every heap buffer, by capacity (allocation, not use).
    std::size_t store_bytes() const override {
        return sizeof(*this)
             + (counts_.capacity() + row_counts_.capacity() + col_counts_.capacity()) * sizeof(double)
             + (cuts_x_.capacity() + cuts_y_.capacity()) * sizeof(double)
             + calibration_.capacity() * sizeof(std::pair<double, double>);
    }
    const char* name() const override { return name_; }
    bool ready() const override { return calibrated_ && total_ > 0.0; }
    void set_neff(double neff) override { lambda_ = (neff - 1.0) / (neff + 1.0); }

private:
    void start() {
        if (!config_.normal_score_inputs) { calibration_.reserve(config_.calibration); return; }
        cuts_x_.resize(config_.bins - 1);
        for (std::size_t k = 1; k < config_.bins; ++k)
            cuts_x_[k - 1] = special::probit(static_cast<double>(k) / static_cast<double>(config_.bins));
        cuts_y_ = cuts_x_;
        calibrated_ = true;
    }

    void build_cutpoints() {
        std::vector<double> xs, ys;
        xs.reserve(calibration_.size());
        ys.reserve(calibration_.size());
        for (const auto& p : calibration_) { xs.push_back(p.first); ys.push_back(p.second); }
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

    static std::size_t bin_of(const std::vector<double>& cuts, double value) {
        return static_cast<std::size_t>(std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
    }

    void insert(double x, double y) {
        const std::size_t i = bin_of(cuts_x_, x), j = bin_of(cuts_y_, y), m = config_.bins;
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
    }

    Config config_;
    const char* name_;
    bool calibrated_ = false;
    double total_ = 0.0;
    double lambda_ = config_.lambda;
    std::vector<double> counts_, row_counts_, col_counts_, cuts_x_, cuts_y_;
    std::vector<std::pair<double, double>> calibration_;
};
