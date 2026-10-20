#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Bounded-window streaming rank-PIT for one marginal, then probit to normal scores.
// See section 0.5 of notes/studies/kalman_changepoint_results.md.
class RankPit {
public:
    explicit RankPit(std::size_t window = 500) : window_(window) {
        sorted_.reserve(window + 1);
        fifo_.reserve(window + 1);
    }

    double transform(double value) {
        const auto lo = std::lower_bound(sorted_.begin(), sorted_.end(), value);
        const double below = static_cast<double>(lo - sorted_.begin());
        const double n = static_cast<double>(sorted_.size());

        // (r + 1/2) / (n + 1) keeps u strictly inside (0,1) so the probit stays finite.
        const double u = (n > 0.0) ? (below + 0.5) / (n + 1.0) : 0.5;

        sorted_.insert(lo, value);
        fifo_.push_back(value);
        if (fifo_.size() > window_) {
            const double old = fifo_.front();
            fifo_.erase(fifo_.begin());
            const auto it = std::lower_bound(sorted_.begin(), sorted_.end(), old);
            if (it != sorted_.end() && *it == old) sorted_.erase(it);
        }
        return u;
    }

    void reset() {
        sorted_.clear();
        fifo_.clear();
    }

    std::size_t store_bytes() const {
        return sizeof(*this) + (sorted_.capacity() + fifo_.capacity()) * sizeof(double);
    }

    std::size_t size() const { return sorted_.size(); }

    // Acklam's rational approximation to the normal quantile; abs error < 1.15e-9.
    static double probit(double p) {
        static constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                        -2.759285104469687e+02, 1.383577518672690e+02,
                                        -3.066479806614716e+01, 2.506628277459239e+00};
        static constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                        -1.556989798598866e+02, 6.680131188771972e+01,
                                        -1.328068155288572e+01};
        static constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                        -2.400758277161838e+00, -2.549732539343734e+00,
                                        4.374664141464968e+00, 2.938163982698783e+00};
        static constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                        2.445134137142996e+00, 3.754408661907416e+00};
        const double pl = 0.02425;
        p = std::clamp(p, 1e-12, 1.0 - 1e-12);
        if (p < pl) {
            const double q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        }
        if (p > 1.0 - pl) {
            const double q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        }
        const double q = p - 0.5, r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }

private:
    std::size_t window_;
    std::vector<double> sorted_;
    std::vector<double> fifo_;
};
