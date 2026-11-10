#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "estimator.hpp"

// Gregory (2019) bivariate empirical-copula summary with exponential forgetting (forward decay).
// A weighted Greenwald-Khanna summary S1 of x whose tuples each carry a weighted GK summary of the y's
// they cover; observation t has weight kappa^{-t}, so relative weights are exponential forgetting with
// N_eff = (1 + kappa) / (1 - kappa). Tuples combine every floor(1/(2 eps)) inserts while their weight
// range stays within 2 eps W (Appendix A.1.2, weighted). estimate() is the Spearman's rho of the
// summarised copula, from rank-interval midpoints (the probe in budget_mpl.md §22 showed the paper's
// r_max queries are biased). Design and results: notes/studies/budget_mpl.md §23.
class GregoryCopula : public DependenceEstimator {
public:
    struct Config {
        double kappa = 0.99;     // retention per observation; 0 or 1 = cumulative (no forgetting)
        double eps = 0.02;       // GK approximation parameter
    };

    GregoryCopula(Config c, const char* name) : c_(c), name_(name) {
        every_ = std::max<std::size_t>(1, static_cast<std::size_t>(1.0 / (2.0 * c_.eps)));
    }

    void update(double x, double y) override {
        const double w = scale_;
        insert_x(x, y, w);
        insert(ys_, y, w);
        W_ += w;
        ++n_;
        if (n_ % every_ == 0) {
            combine();
            compress(ys_, 2.0 * c_.eps * W_);
        }
        if (c_.kappa > 0.0 && c_.kappa < 1.0) {
            scale_ /= c_.kappa;
            if (scale_ > 1e150) renormalise(1e-150);
        }
        dirty_ = true;
    }

    double estimate() const override {
        if (!dirty_) return rho_;
        dirty_ = false;
        if (n_ < 30 || W_ <= 0.0) return rho_ = 0.0;
        cum_y_.resize(ys_.size());
        double c = 0.0;
        for (std::size_t k = 0; k < ys_.size(); ++k) { c += ys_[k].g; cum_y_[k] = c; }
        // A tuple with weight g covers the values in (v_prev, v]: its rank is the midpoint of that
        // interval (widened by Delta for S1), the first tuple of each summary covers only its own value.
        double acc = 0.0, cum = 0.0;
        for (std::size_t i = 0; i < x_.size(); ++i) {
            const Tuple& t = x_[i];
            const double u1 = (cum + 0.5 * (t.g + t.d)) / W_ - 0.5;
            cum += t.g;
            const Summary& s = sub_[i];
            double mass = 0.0;
            for (const Tuple& e : s) mass += e.g;
            if (mass <= 0.0) continue;
            double inner = 0.0, prev = 0.0;
            for (std::size_t k = 0; k < s.size(); ++k) {
                const double r = rank_y(s[k].v);
                const double r2 = (k == 0) ? r : 0.5 * (prev + r);
                prev = r;
                inner += s[k].g * (r2 / W_ - 0.5);
            }
            acc += t.g * u1 * inner / mass;
        }
        return rho_ = std::clamp(12.0 * acc / W_, -1.0, 1.0);
    }

    void reset() override {
        x_.clear(); sub_.clear(); ys_.clear();
        W_ = 0.0; scale_ = 1.0; n_ = 0; rho_ = 0.0; dirty_ = true;
    }

    std::size_t store_bytes() const override {
        std::size_t tuples = x_.size() + ys_.size();
        for (const auto& s : sub_) tuples += s.size();
        return sizeof(*this) + tuples * sizeof(Tuple) + sub_.size() * sizeof(Summary);
    }

    std::size_t stored_tuples() const {
        std::size_t t = x_.size() + ys_.size();
        for (const auto& s : sub_) t += s.size();
        return t;
    }

    const char* name() const override { return name_; }
    bool ready() const override { return n_ >= 30; }

private:
    struct Tuple { double v, g, d, w; };   // value, weight covered, rank-weight uncertainty, v's own weight
    using Summary = std::vector<Tuple>;

    // Weighted GK insert: Delta = g + Delta - w of the right neighbour (w = its own item's weight, the
    // paper's -1); without the own weight, fresh heavy neighbours give old tuples a Delta they never lose.
    static void insert(Summary& s, double v, double w) {
        auto it = std::lower_bound(s.begin(), s.end(), v, [](const Tuple& t, double x) { return t.v < x; });
        const double d = (it == s.begin() || it == s.end()) ? 0.0 : std::max(0.0, it->g + it->d - it->w);
        s.insert(it, Tuple{v, w, d, w});
    }

    void insert_x(double x, double y, double w) {
        auto it = std::lower_bound(x_.begin(), x_.end(), x, [](const Tuple& t, double v) { return t.v < v; });
        const std::size_t i = static_cast<std::size_t>(it - x_.begin());
        const double d = (i == 0 || i == x_.size()) ? 0.0 : std::max(0.0, x_[i].g + x_[i].d - x_[i].w);
        x_.insert(it, Tuple{x, w, d, w});
        sub_.insert(sub_.begin() + static_cast<std::ptrdiff_t>(i), Summary{Tuple{y, w, 0.0, w}});
    }

    // Merge neighbours (first and last kept) while g_i + g_{i+1} + d_{i+1} <= cap.
    static void compress(Summary& s, double cap) {
        if (s.size() < 3) return;
        for (std::size_t i = s.size() - 2; i >= 1; --i) {
            if (s[i].g + s[i + 1].g + s[i + 1].d <= cap) {
                s[i + 1].g += s[i].g;
                s.erase(s.begin() + static_cast<std::ptrdiff_t>(i));
            }
            if (i == 1) break;
        }
    }

    // Weighted Greenwald-Khanna (2004) merge, via cumulative rmin / rmax (Appendix A.2).
    static Summary merge(const Summary& a, const Summary& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        struct R { double v, rmin, rmax, w; };
        const auto ranks = [](const Summary& s) {
            std::vector<R> r(s.size());
            double cum = 0.0;
            for (std::size_t i = 0; i < s.size(); ++i) { cum += s[i].g; r[i] = {s[i].v, cum, cum + s[i].d, s[i].w}; }
            return r;
        };
        const auto ra = ranks(a), rb = ranks(b);
        std::vector<R> out;
        out.reserve(ra.size() + rb.size());
        for (int pass = 0; pass < 2; ++pass) {
            const auto& p = pass ? rb : ra;
            const auto& q = pass ? ra : rb;
            for (const R& e : p) {
                auto it = std::upper_bound(q.begin(), q.end(), e.v, [](double v, const R& t) { return v < t.v; });
                const double qmin = (it == q.begin()) ? 0.0 : std::prev(it)->rmin;
                const double qmax = (it == q.end()) ? (it == q.begin() ? 0.0 : std::prev(it)->rmax) : it->rmax - it->w;
                out.push_back({e.v, e.rmin + qmin, e.rmax + std::max(qmin, qmax), e.w});
            }
        }
        std::sort(out.begin(), out.end(), [](const R& l, const R& r) { return l.v < r.v; });
        Summary m(out.size());
        double prev = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            const double rmin = std::max(out[i].rmin, prev);
            m[i] = {out[i].v, rmin - prev, std::max(0.0, out[i].rmax - rmin), out[i].w};
            prev = rmin;
        }
        return m;
    }

    // Appendix A.1.2, weighted: from the right, absorb left neighbours while the run stays within 2 eps W.
    void combine() {
        const double cap = 2.0 * c_.eps * W_;
        if (x_.size() < 3) return;
        std::size_t j = x_.size() - 1;
        while (j > 1) {
            std::size_t k = 0;
            double tot = x_[j].g + x_[j].d;
            while (j - k - 1 >= 1 && tot + x_[j - k - 1].g < cap) { ++k; tot += x_[j - k].g; }
            if (k >= 1) {
                Summary q = sub_[j - k];
                double g = x_[j - k].g;
                for (std::size_t t = j - k + 1; t <= j; ++t) { q = merge(q, sub_[t]); g += x_[t].g; }
                double mass = 0.0;
                for (const Tuple& e : q) mass += e.g;
                compress(q, 2.0 * c_.eps * mass);
                const Tuple keep{x_[j].v, g, x_[j].d, x_[j].w};
                x_.erase(x_.begin() + static_cast<std::ptrdiff_t>(j - k), x_.begin() + static_cast<std::ptrdiff_t>(j + 1));
                x_.insert(x_.begin() + static_cast<std::ptrdiff_t>(j - k), keep);
                sub_.erase(sub_.begin() + static_cast<std::ptrdiff_t>(j - k), sub_.begin() + static_cast<std::ptrdiff_t>(j + 1));
                sub_.insert(sub_.begin() + static_cast<std::ptrdiff_t>(j - k), std::move(q));
            }
            if (j < k + 2) break;
            j = j - k - 1;
        }
    }

    // Midpoint weighted count of y values <= v (Appendix A.4 with the rank interval's midpoint).
    double rank_y(double v) const {
        const auto it = std::upper_bound(ys_.begin(), ys_.end(), v, [](double x, const Tuple& t) { return x < t.v; });
        const std::size_t k = static_cast<std::size_t>(it - ys_.begin());
        const double below = (k == 0) ? 0.0 : cum_y_[k - 1];
        if (k == 0 || k == ys_.size()) return below;
        // v lies in (v_{k-1}, v_k): somewhere among the g_k values tuple k covers.
        return below + 0.5 * (ys_[k].g + ys_[k].d);
    }

    void renormalise(double f) {
        for (auto& t : x_) { t.g *= f; t.d *= f; t.w *= f; }
        for (auto& s : sub_) for (auto& t : s) { t.g *= f; t.d *= f; t.w *= f; }
        for (auto& t : ys_) { t.g *= f; t.d *= f; t.w *= f; }
        W_ *= f;
        scale_ *= f;
    }

    Config c_;
    const char* name_;
    std::size_t every_ = 1;
    Summary x_;                  // S1
    std::vector<Summary> sub_;   // S2^i, aligned with x_
    Summary ys_;                 // weighted GK summary of all y (for y-ranks)
    double W_ = 0.0, scale_ = 1.0;
    std::size_t n_ = 0;
    mutable std::vector<double> cum_y_;
    mutable double rho_ = 0.0;
    mutable bool dirty_ = true;
};
