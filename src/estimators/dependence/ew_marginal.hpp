#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

// N_eff,m of every bank's and detector's shared EW marginal: 500 unless BANK_MARGINAL_NEFF overrides it
// (the sensitivity runs at 1999). Separate from any estimator's dependence memory.
inline double default_marginal_neff() {
    static const double v = [] {
        const char* e = std::getenv("BANK_MARGINAL_NEFF");
        return e ? std::atof(e) : 500.0;
    }();
    return v;
}

// Exponentially weighted empirical CDF of one marginal, F_n(x) = sum_i a_{n,i} 1{X_i <= x} with
// a_{n,i} proportional to lambda^{n-i} (thesis 5.1.2 applied to indicators), held in bounded memory as
// a weighted Greenwald-Khanna summary with forward decay: observation i enters with weight
// lambda^{-i}, so relative weights are exact without rescaling old tuples. Combining every
// floor(1/(2 eps)) inserts keeps g + Delta <= 2 eps W. Ranks are read at the midpoint of each rank
// interval. neff <= 0 gives no forgetting. Design and checks: notes/studies/ew_marginals.md.
class EwMarginal {
public:
    explicit EwMarginal(double neff = 500.0, double eps = 0.005)
        : kappa_(neff > 1.0 ? (neff - 1.0) / (neff + 1.0) : 1.0), eps_(eps),
          every_(std::max<std::size_t>(1, static_cast<std::size_t>(1.0 / (2.0 * eps)))) {}

    // Pseudo-observation of x from the state BEFORE x, (B + w)/(W + 2w) with B the weight below x and w
    // the weight x is about to receive: (R + 1)/(n + 2) when nothing is forgotten. Then absorbs x.
    double transform(double x) {
        const double w = scale_;
        const double u = (below(x) + w) / (W_ + 2.0 * w);
        insert(x, w);
        return u;
    }

    double cdf(double x) const { return W_ > 0.0 ? below(x) / W_ : 0.5; }

    std::size_t size() const { return n_; }
    std::size_t tuples() const { return s_.size(); }
    std::size_t heap_bytes() const { return s_.capacity() * sizeof(Tuple); }
    std::size_t store_bytes() const { return sizeof(*this) + heap_bytes(); }

    void reset() { s_.clear(); W_ = 0.0; scale_ = 1.0; n_ = 0; }

private:
    struct Tuple { double v, g, d, w; };   // value, weight covered, rank uncertainty, v's own weight

    // Weight strictly below x: whole tuples below, plus the midpoint of the covered weight of the
    // first tuple at or above x that may lie below it, [0, g + Delta - w].
    double below(double x) const {
        double cum = 0.0;
        for (const Tuple& t : s_) {
            if (t.v >= x) return cum + 0.5 * std::max(0.0, t.g + t.d - t.w);
            cum += t.g;
        }
        return cum;
    }

    void insert(double x, double w) {
        auto it = std::lower_bound(s_.begin(), s_.end(), x, [](const Tuple& t, double v) { return t.v < v; });
        // Interior Delta = g + Delta - w of the right neighbour: the weighted form of GK's "-1".
        const double d = (it == s_.begin() || it == s_.end()) ? 0.0 : std::max(0.0, it->g + it->d - it->w);
        s_.insert(it, Tuple{x, w, d, w});
        W_ += w;
        if (++n_ % every_ == 0) compress();
        if (kappa_ < 1.0) {
            scale_ /= kappa_;
            if (scale_ > 1e150) rescale(1e-150);   // forward-decay weights grow without bound
        }
    }

    // GK compress (first and last tuples kept): merge neighbours while g_i + g_{i+1} + Delta_{i+1} <= 2 eps W.
    void compress() {
        const double cap = 2.0 * eps_ * W_;
        if (s_.size() < 3) return;
        for (std::size_t i = s_.size() - 2; i >= 1; --i) {
            if (s_[i].g + s_[i + 1].g + s_[i + 1].d <= cap) {
                s_[i + 1].g += s_[i].g;
                s_.erase(s_.begin() + static_cast<std::ptrdiff_t>(i));
            }
            if (i == 1) break;
        }
    }

    void rescale(double f) {
        for (Tuple& t : s_) { t.g *= f; t.d *= f; t.w *= f; }
        W_ *= f;
        scale_ *= f;
    }

    double kappa_, eps_;
    std::size_t every_;
    std::vector<Tuple> s_;
    double W_ = 0.0, scale_ = 1.0;
    std::size_t n_ = 0;
};
