#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "estimator.hpp"
#include "kll_sketch.hpp"
#include "../../special_functions.hpp"
#if __has_include(<kll_sketch.hpp>)   // DataSketches on the include path (experiments/build.sh, Makefile)
#include "datasketches_kll.hpp"
#define RECURSIVE_MPL_HAS_DS 1
#endif

// Recursive maximum pseudo-likelihood for the Gaussian copula (RMPL, thesis 4.3). Pseudo-observations come from
// the margin state BEFORE the pair is inserted:
//   ExactNormal  inputs are already normal scores z (the thesis arms: z = probit of shared KLL pseudo-observations)
//   Kll / DsKll  its own cumulative KLL sketch per margin (hand-written / Apache DataSketches), capacity kll_k
// Updates, gain g = 2/(N_eff + 1):
//   ConstantGain r <- clip(r + g_m s(r) / I(r)), start-up g_m = g / (1 - (1 - g)^m) (EW weights normalised)
//   Exact        arg max of sum_k (1 - g)^k log c_r(a_k, b_k): the root in [-1, 1] of -W r^3 + P r^2 + (W - S) r + P
class RecursiveMPL : public DependenceEstimator {
public:
    enum class Margin { Kll, DsKll, ExactNormal };
    enum class Update { ConstantGain, Exact };

    struct Config {
        Margin margin = Margin::Kll;
        Update update = Update::ConstantGain;
        double gain = 0.01;
        std::size_t kll_k = 200;
        double delta = 1e-3;            // pseudo-observations clipped to [delta, 1 - delta]
        double r_max = 0.999;
        std::size_t warmup = 30;        // margin observations before the copula updates
        unsigned seed = 1;
    };

    RecursiveMPL(Config c, const char* name) : c_(c), name_(name) { make_margins(); }

    void update(double x, double y) override {
        double u, v;
        if (c_.margin == Margin::ExactNormal) {
            u = 0.5 * std::erfc(-x / std::sqrt(2.0));
            v = 0.5 * std::erfc(-y / std::sqrt(2.0));
        } else if (kx_) {
            u = kx_->cdf(x); v = ky_->cdf(y);
            kx_->insert(x); ky_->insert(y);
        } else {
#ifdef RECURSIVE_MPL_HAS_DS
            u = dx_->cdf(x); v = dy_->cdf(y);
            dx_->insert(x); dy_->insert(y);
#else
            u = v = 0.5;
#endif
        }
        const std::size_t seen = n_++;
        if (seen < c_.warmup) return;
        u = std::clamp(u, c_.delta, 1.0 - c_.delta);
        v = std::clamp(v, c_.delta, 1.0 - c_.delta);
        const double z1 = special::probit(u), z2 = special::probit(v);
        if (c_.update == Update::Exact) { step_exact(z1, z2); return; }
        const double r = r_, q = 1.0 - r * r;
        const double s = r / q + ((1.0 + r * r) * z1 * z2 - r * (z1 * z1 + z2 * z2)) / (q * q);
        const double info = (1.0 + r * r) / (q * q);
        ++m_;
        wsum_ = (1.0 - gain_) * wsum_ + gain_;
        r_ = std::clamp(r + gain_ / wsum_ * s / info, -c_.r_max, c_.r_max);
    }

    double estimate() const override { return (6.0 / M_PI) * std::asin(r_ / 2.0); }

    void reset() override {
        n_ = m_ = 0; r_ = 0.0; wsum_ = 0.0; gain_ = c_.gain; W_ = S_ = P_ = 0.0;
        make_margins();
    }

    // The object plus the heap of its margin sketches.
    std::size_t store_bytes() const override {
        std::size_t b = sizeof(*this);
        if (kx_) b += kx_->store_bytes() + ky_->store_bytes();
#ifdef RECURSIVE_MPL_HAS_DS
        if (dx_) b += dx_->store_bytes() + dy_->store_bytes();
#endif
        return b;
    }
    const char* name() const override { return name_; }
    void set_neff(double neff) override { gain_ = 2.0 / (neff + 1.0); }
    bool ready() const override { return m_ > 0 || n_ > c_.warmup; }

private:
    // f(-1) = S + 2P >= 0 and f(1) = 2P - S <= 0, so a root exists; Newton from the previous root inside a
    // shrinking bracket (bisection when Newton leaves it).
    void step_exact(double a, double b) {
        const double kappa = 1.0 - c_.gain;
        W_ = kappa * W_ + 1.0;
        S_ = kappa * S_ + a * a + b * b;
        P_ = kappa * P_ + a * b;
        const auto f = [&](double r) { return -W_ * r * r * r + P_ * r * r + (W_ - S_) * r + P_; };
        const auto df = [&](double r) { return -3.0 * W_ * r * r + 2.0 * P_ * r + (W_ - S_); };
        double lo = -1.0, hi = 1.0, r = r_;
        for (int it = 0; it < 60; ++it) {
            const double fr = f(r);
            if (fr > 0.0) lo = r; else hi = r;
            const double d = df(r);
            double nr = (d != 0.0) ? r - fr / d : 0.5 * (lo + hi);
            if (!(nr > lo && nr < hi)) nr = 0.5 * (lo + hi);
            if (std::abs(nr - r) < 1e-12) { r = nr; break; }
            r = nr;
        }
        r_ = std::clamp(r, -c_.r_max, c_.r_max);
    }

    // Fresh sketches; DataSketches' coin generator is reseeded so a run is reproducible.
    void make_margins() {
        if (c_.margin == Margin::Kll) {
            kx_ = std::make_unique<KllSketch>(c_.kll_k, 2 * c_.seed + 1);
            ky_ = std::make_unique<KllSketch>(c_.kll_k, 2 * c_.seed + 2);
        }
#ifdef RECURSIVE_MPL_HAS_DS
        if (c_.margin == Margin::DsKll) {
            datasketches_kll::seed(c_.seed);
            dx_ = std::make_unique<datasketches_kll::Kll>(static_cast<std::uint16_t>(c_.kll_k));
            dy_ = std::make_unique<datasketches_kll::Kll>(static_cast<std::uint16_t>(c_.kll_k));
        }
#endif
    }

    Config c_;
    const char* name_;
    std::size_t n_ = 0, m_ = 0;   // margin observations, copula updates
    double r_ = 0.0;
    double wsum_ = 0.0;           // accumulated EW weight, for the normalised start-up gain
    double gain_ = c_.gain;       // current gain (moved only by set_neff)
    double W_ = 0.0, S_ = 0.0, P_ = 0.0;   // Exact
    std::unique_ptr<KllSketch> kx_, ky_;
#ifdef RECURSIVE_MPL_HAS_DS
    std::unique_ptr<datasketches_kll::Kll> dx_, dy_;
#endif
};
