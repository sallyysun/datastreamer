#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "estimator.hpp"

class WindowedRankEstimator : public DependenceEstimator {
public:
    explicit WindowedRankEstimator(std::size_t window = 256)
        : capacity_(window) {
        xs_.reserve(window);
        ys_.reserve(window);
    }

    void update(double x, double y) override {
        if (xs_.size() < capacity_) {
            xs_.push_back(x);
            ys_.push_back(y);
        } else {
            xs_[next_] = x;
            ys_[next_] = y;
            next_ = (next_ + 1) % capacity_;
        }
    }

    double estimate() const override {
        const std::size_t n = xs_.size();
        if (n < 2) return 0.0;

        const std::vector<double> rx = average_ranks(xs_);
        const std::vector<double> ry = average_ranks(ys_);

        const double mean = static_cast<double>(n + 1.0) / 2.0;
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

    double kendall() const {
        const std::size_t n = xs_.size();
        if (n < 2) return 0.0;

        std::size_t concordant = 0, discordant = 0;
        for (std::size_t i = 0; i < n - 1; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double dx = xs_[i] - xs_[j];
                const double dy = ys_[i] - ys_[j];
                if (dx == 0.0 && dy == 0.0) { ++ties_x; ++ties_y; continue; }
                if (dx == 0.0) { ++ties_x; continue; }
                if (dy == 0.0) { ++ties_y; continue; }
                if (dx * dy > 0.0) ++concordant; else ++discordant;
            }
        }

        const double n0 = 0.5 * static_cast<double>(n) * (static_cast<double>(n) - 1.0);
        const double denom = std::sqrt((n0 - static_cast<double>(ties_x)) * (n0 - static_cast<double>(ties_y)));
        return (denom > 0.0) ? (static_cast<double>(concordant - discordant) / denom) : 0.0;
    }

    std::size_t state_bytes() const override {
        return 2 * capacity_ * sizeof(double);
    }

    const char* name() const override { return "windowed_rank"; }

    std::size_t count() const { return xs_.size(); }

private:
    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&v](std::size_t a, size_t b) -> bool { return v[a] < v[b]; });

        std::vector<double> ranks(n);
        std::size_t i = 0;
        while(i < n) {
            std::size_t j = i;
            while (j + 1 < n && v[order[j + 1]] == v[order[i]]) ++j;
            const double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
            for (std:size_t k = i; k <= j; ++k) ranks[order[k]] = avg;
            i = j + 1;
        }
        return ranks;
    }

    std::size_t capacity_;
    std::size_t next_ = 0;
    std::vector<double> xs_;
    std::vector<double> ys_;
};