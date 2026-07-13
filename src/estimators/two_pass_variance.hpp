#pragma once

#include <cstddef>
#include <vector>

class TwoPassVarianceEstimator {
public:
    explicit TwoPassVarianceEstimator(std::size_t window = 256) : capacity_(window) {
        values_.reserve(window);
    }

    void update(double x) {
        if (values_.size() < capacity_) {
            values_.push_back(x);
        } else {
            values_[next_] = x;
        }
        next_ = (next_ + 1) % capacity_;
    }

    double mean() const {
        if (values_.empty()) return 0.0;
        double sum = 0.0;
        for (double v : values_) sum += v;
        return sum / static_cast<double>(values_.size());
    }

    double variance() const {
        const std::size_t n = values_.size();
        if (n < 2) return 0.0;
        const double m = mean();
        double sum_sq_diff = 0.0;
        for (double v : values_) sum_sq_diff += (v - m) * (v - m);
        return sum_sq_diff / static_cast<double>(values_.size() - 1);
    }

    std::size_t state_bytes() const {
        return capacity_ * sizeof(double);
    }

    std::size_t count() const {
        return values_.size();
    }

private:
    std::size_t capacity_;
    std::size_t next_ = 0;
    std::vector<double> values_;
};