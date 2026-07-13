#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "estimator.hpp"

class XiaoSpearmanEstimator : public DependenceEstimator {
public:
    struct Config {
        std::size_t bins = 30;
        std::size_t window = 0;
        std::size_t calibration = 300;
    };

    XiaoSpearmanEstimator(Config config, const char* name)
        : config_(config),
        name_(name),
        counts_(config.bins * config.bins, 0),
        row_counts_(config.bins, 0),
        col_counts_(config.bins, 0) {
            calibration_.reserve(config.calibration);
        }

    void update(double x, double y) override {
        if (!calibrated_) {
            calibration_.push_back({x, y});
            if (calibration_.size() >= config_.calibration) {
                build_cutpoints();
                for (const auto& pair : pair<double, double> const & : calibration_) {
                    insert(pair.first, pair.second);
                }
                calibration_.clear();
                calibration_.shrink_to_fit();
                calibrated_ = true;
            }
            return;
        }
        insert(x, y);
    }

    double estimate() const override {
        if (!calibrated_ || n_ < 2) return 0.0;

        const std::size_t m = config_.bins;
        const double center = (static_cast<double>(n_) + 1.0) / 2.0;

        std::vector<double> row_rank(m), col_rank(m);
        double norm_row = 0.0, norm_col = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            row_rank[i] = static_cast<double>(before) + (row_counts_[i] + 1.0) / 2.0 - center;
            norm_row += row_counts_[i] * row_rank[i] * row_rank[i];
            before += row_counts_[i];
        }
        before = 0;
        for (std::size_t j = 0; j < m; ++j) {
            col_rank[j] = static_cast<double>(before) + (col_counts_[j] + 1.0) / 2.0 - center;
            norm_col += col_counts_[j] * col_rank[j] * col_rank[j];
            before += col_counts_[j];
        }

        const double demon = std::sqrt(norm_row * norm_col);
        if (demon <= 0.0) return 0.0;

        double bilinear = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                const long long c = counts_[i * m + j];
                if (c) bilinear += c * row_rank[i] * col_rank[j];
            }
        }
        return std::clamp(bilinear / demon, -1.0, 1.0);
    }

    std::size_t store_bytes() const override {
        return counts_.size() * sizeof(std::uint32_t) 
            + (row_counts_.size() + col_counts_.size()) * sizeof(std::uint32_t)
            + (cuts_x_.size() + cuts_y_.size()) * sizeof(double)
            + window_.size() * sizeof(BinPair)
            + calibration_.capacity() * sizeof(std::pair<double, double>);
    }

    const char* name() const override {
        return name_;
    }

private:
    struct BinPair {
        std::uint16_t i;
        std::uint16_t j;
    };

    void build_cutpoints() {
        std::vector<double> xs, ys;
        xs.reserve(calibration_.size());
        ys.reserve(calibration_.size());
        for (const auto& pair : pair<double, double> const & : calibration_) {
            xs.push_back(pair.first);
            ys.push_back(pair.second);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        cuts_x_.resize(config_.bins - 1);
        cuts_y_.resize(config_.bins - 1);
        for (std::size_t k = 0; k < config_.bins; ++k) {
            const std::size_t idx = k * xs.size() / config_.bins;
            cuts_x_[k] = xs[std::min(idx, xs.size() - 1)];
            cuts_y_[k] = ys[std::min(idx, ys.size() - 1)];
        }
    }

    std::size_t bin_of(const std::vector<double>& cuts, double value) const {
        return static_cast<std::size_t>(std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
    }

    void insert(double x, double y) {
        const std::size_t i = bin_of(cuts_x_, x);
        const std::size_t j = bin_of(cuts_y_, y);
        ++counts_[i * config_.bins + j];
        ++row_counts_[i];
        ++col_counts_[j];
        ++n_;
        if (config_.window > 0) {
            window_.push_back({static_cast<std::uint16_t>(i), static_cast<std::uint16_t>(j)});
            if (window_.size() > config_.window) {
                const BinPair& old = window_.front();
                window_.pop_front();
                --counts_[old.i * config_.bins + old.j];
                --row_counts_[old.i];
                --col_counts_[old.j];
                --n_;
            }
        }
    }