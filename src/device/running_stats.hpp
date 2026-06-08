#pragma once

#include <cmath>
#include <cstddef>

// Online mean/variance via Welford's algorithm. O(1) per update, no need to
// keep the samples around -- this is the kind of estimator the StateManager can
// maintain incrementally as each event arrives.
class RunningStats {
public:
    void add(double x) {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        const double delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    std::size_t count() const { return n_; }

    double mean() const { return n_ > 0 ? mean_ : 0.0; }

    // sample variance (Bessel-corrected)
    double variance() const {
        return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0;
    }

    double stddev() const { return std::sqrt(variance()); }

private:
    std::size_t n_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};
