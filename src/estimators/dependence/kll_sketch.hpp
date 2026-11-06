#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// KLL quantile sketch (Karnin, Lang & Liberty 2016) for one marginal: approximate
// ranks in bounded memory. Cumulative, so exact only for stationary margins.
// See notes/studies/budget_mpl.md.
class KllSketch {
public:
    explicit KllSketch(std::size_t k = 200, unsigned seed = 1) : k_(k), rng_(seed) {
        levels_.emplace_back();
    }

    void insert(double x) {
        levels_[0].push_back(x);
        ++n_;
        ++retained_;
        if (retained_ > total_capacity()) compress();
    }

    // Weighted count of retained items strictly below x, over n: F_hat(x-).
    double cdf(double x) const {
        if (n_ == 0) return 0.5;
        double w = 0.0;
        for (std::size_t h = 0; h < levels_.size(); ++h) {
            std::size_t c = 0;
            for (double v : levels_[h]) c += (v < x);
            w += std::ldexp(static_cast<double>(c), static_cast<int>(h));
        }
        return w / static_cast<double>(n_);
    }

    std::size_t n() const { return n_; }
    std::size_t retained() const { return retained_; }

    std::size_t store_bytes() const {
        std::size_t b = sizeof(*this) + levels_.capacity() * sizeof(std::vector<double>);
        for (const auto& l : levels_) b += l.capacity() * sizeof(double);
        return b;
    }

    void reset() {
        levels_.assign(1, {});
        n_ = retained_ = 0;
    }

private:
    std::size_t capacity(std::size_t h) const {
        const std::size_t depth = levels_.size() - 1 - h;
        return std::max<std::size_t>(
            2, static_cast<std::size_t>(std::ceil(static_cast<double>(k_) * std::pow(2.0 / 3.0, depth))));
    }
    std::size_t total_capacity() const {
        std::size_t c = 0;
        for (std::size_t h = 0; h < levels_.size(); ++h) c += capacity(h);
        return c;
    }

    void compress() {
        for (std::size_t h = 0; h < levels_.size(); ++h) {
            if (levels_[h].size() < capacity(h)) continue;
            if (h + 1 == levels_.size()) levels_.emplace_back();
            auto& lv = levels_[h];
            std::sort(lv.begin(), lv.end());
            // An odd item stays behind so the promoted half carries exact weight.
            double keep = 0.0;
            const bool odd = lv.size() % 2 == 1;
            if (odd) { keep = lv.back(); lv.pop_back(); }
            const std::size_t off = coin_(rng_);
            for (std::size_t i = off; i < lv.size(); i += 2) levels_[h + 1].push_back(lv[i]);
            retained_ -= lv.size() / 2;
            lv.clear();
            if (odd) lv.push_back(keep);
            return;
        }
    }

    std::size_t k_;
    std::vector<std::vector<double>> levels_;
    std::size_t n_ = 0;
    std::size_t retained_ = 0;
    std::minstd_rand rng_;   // 8 bytes: an mt19937 would outweigh a small sketch
    std::uniform_int_distribution<std::size_t> coin_{0, 1};
};
