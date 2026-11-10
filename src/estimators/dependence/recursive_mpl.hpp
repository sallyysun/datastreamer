#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "estimator.hpp"
#include "memory_control.hpp"
#include "predictive_density.hpp"
#if __has_include(<kll_sketch.hpp>)   // DataSketches on the include path (experiments/Makefile DS_INC)
#include "ds_margin.hpp"
#define RECURSIVE_MPL_HAS_DS 1
#endif
#include "ew_marginal.hpp"
#include "kll_sketch.hpp"
#include "marginal_rank_pit.hpp"
#include "parametric_copula.hpp"

// Recursive pseudo-likelihood for the Gaussian copula. Pseudo-observations come from the marginal state
// BEFORE the current pair is inserted. Update modes (kappa = 1 - gain):
//   ConstantGain   r <- clip(r + gain I(r)^-1 s(r), +-r_max)                       (spec of 2026-11-06 a.m.)
//   FisherTheta    thesis 4.3.2: theta = atanh r, H <- kappa H + (1 + r^2), H_0 = 1,
//                  theta <- Pi[theta + s_theta / H] with |r| <= r_max; no step clamp     (current thesis)
//   Exact          arg max of Q_t(r) = sum kappa^k log c_r(a_k, b_k): the root in (-1, 1) of
//                  -W r^3 + P r^2 + (W - S) r + P, W = sum kappa^k, S = sum kappa^k (a^2+b^2), P = sum kappa^k ab
//   CopulaFit      the pre-2026-11-06 registered recursion (step clamp, unprojected phi)
// Spec and results: notes/studies/budget_mpl.md.
class RecursiveMPL : public DependenceEstimator, public PredictiveDensity, public MemoryControlled {
public:
    enum class Margin { Kll, DsKll, ExactNormal, RankWindow, Ew };
    enum class Update { ConstantGain, FisherTheta, Exact, CopulaFit };

    struct Config {
        Margin margin = Margin::Kll;
        Update update = Update::ConstantGain;
        double gain = 0.01;
        std::size_t kll_k = 200;
        std::size_t window = 500;
        double marginal_neff = 500.0;   // Margin::Ew: EW empirical CDF (ew_marginal.hpp)
        double delta = 1e-3;
        double r_max = 0.999;
        std::size_t warmup = 30;   // marginal observations before the copula updates
        // Start-up gain on the m-th copula update. startup_normalised: g_m = gain / (1 - (1 - gain)^m),
        // the EW weights of thesis 5.1.3 normalised to sum to one, which is Fisher scoring on the EW
        // pseudo-log-likelihood with L_0 = 0 (g_1 = 1, g_m -> gain). Otherwise max(gain, 1/m), the
        // pre-2026-11-09 rule. startup = false: the constant gain from r_0 = 0.
        bool startup = true;
        bool startup_normalised = true;
        unsigned seed = 1;
    };

    RecursiveMPL(Config c, const char* name)
        : c_(c), name_(name), kx_(c.kll_k, 2 * c.seed + 1), ky_(c.kll_k, 2 * c.seed + 2),
          px_(c.margin == Margin::RankWindow ? c.window : 1),   // an unused RankPit(500) reserves 16 KB
          py_(c.margin == Margin::RankWindow ? c.window : 1),
          fit_(copula::Family::Gaussian, 4.0, 1.0 - c.gain, 0.0, true) {
        if (c_.margin == Margin::DsKll) make_ds();
    }

    // Gaussian copula log density at the current r, for banks that score arms by their own density.
    double log_predictive(double u, double v) const override {
        u = std::clamp(u, c_.delta, 1.0 - c_.delta);
        v = std::clamp(v, c_.delta, 1.0 - c_.delta);
        const double r = (c_.update == Update::CopulaFit) ? fit_.param() : r_;
        return copula::Gaussian::log_density(u, v, r);
    }

    void update(double x, double y) override {
        double u = 0.5, v = 0.5;
        std::size_t seen = n_;
        switch (c_.margin) {
            case Margin::Kll:
                u = kx_.cdf(x); v = ky_.cdf(y);
                kx_.insert(x); ky_.insert(y);
                break;
            case Margin::DsKll:
#ifdef RECURSIVE_MPL_HAS_DS
                u = dx_->cdf(x); v = dy_->cdf(y);
                dx_->insert(x); dy_->insert(y);
#endif
                break;
            case Margin::ExactNormal:
                u = 0.5 * std::erfc(-x / std::sqrt(2.0));
                v = 0.5 * std::erfc(-y / std::sqrt(2.0));
                break;
            case Margin::RankWindow:
                seen = px_.size();
                u = px_.transform(x); v = py_.transform(y);
                break;
            case Margin::Ew:
                seen = ex_.size();
                u = ex_.transform(x); v = ey_.transform(y);
                break;
        }
        ++n_;
        if (seen < c_.warmup) return;
        u = std::clamp(u, c_.delta, 1.0 - c_.delta);
        v = std::clamp(v, c_.delta, 1.0 - c_.delta);
        if (c_.update == Update::CopulaFit) { fit_.update_uv(u, v); return; }
        const double z1 = RankPit::probit(u), z2 = RankPit::probit(v);
        if (c_.update == Update::FisherTheta || c_.update == Update::Exact) { step_discounted(z1, z2); return; }
        const double r = r_, q = 1.0 - r * r;
        const double s = r / q + ((1.0 + r * r) * z1 * z2 - r * (z1 * z1 + z2 * z2)) / (q * q);
        const double info = (1.0 + r * r) / (q * q);
        ++m_;
        double g = gain_;
        if (c_.startup) {
            if (c_.startup_normalised) {
                wsum_ = (1.0 - gain_) * wsum_ + gain_;
                g = gain_ / wsum_;
            } else {
                g = std::max(gain_, 1.0 / static_cast<double>(m_));
            }
        }
        r_ = std::clamp(r + g * s / info, -c_.r_max, c_.r_max);
    }

    double estimate() const override {
        if (c_.update == Update::CopulaFit) return fit_.spearman();
        return (6.0 / M_PI) * std::asin(r_ / 2.0);
    }

    void reset() override {
        n_ = 0; m_ = 0; r_ = 0.0; wsum_ = 0.0; gain_ = c_.gain; theta_ = 0.0; H_ = 1.0; W_ = S_ = P_ = 0.0;
        kx_.reset(); ky_.reset(); px_.reset(); py_.reset(); ex_.reset(); ey_.reset(); fit_.reset();
        if (c_.margin == Margin::DsKll) make_ds();
    }

    // Everything the object holds: itself (every update mode's scalars, both sketch objects, the rank
    // windows), plus the heap of whichever margin is in use (the others hold almost none).
    std::size_t store_bytes() const override {
        std::size_t b = sizeof(*this) + kx_.heap_bytes() + ky_.heap_bytes() + px_.heap_bytes() + py_.heap_bytes()
                      + ex_.heap_bytes() + ey_.heap_bytes();
#ifdef RECURSIVE_MPL_HAS_DS
        if (dx_) b += dx_->store_bytes() + dy_->store_bytes();
#endif
        return b;
    }

    const char* name() const override { return name_; }
    // Constant-gain mode: gain 2/(N + 1).
    void set_neff(double neff) override { gain_ = 2.0 / (neff + 1.0); }
    bool ready() const override { return c_.update == Update::CopulaFit ? n_ > c_.warmup : m_ > 0 || n_ > c_.warmup; }

private:
    void step_discounted(double a, double b) {
        const double kappa = 1.0 - c_.gain;
        if (c_.update == Update::FisherTheta) {
            const double r = r_, q = 1.0 - r * r;
            const double s_r = r / q + ((1.0 + r * r) * a * b - r * (a * a + b * b)) / (q * q);
            H_ = kappa * H_ + (1.0 + r * r);                   // expected information in theta, at theta_{t-1}
            const double tmax = std::atanh(c_.r_max);
            theta_ = std::clamp(theta_ + s_r * q / H_, -tmax, tmax);   // s_theta = s_r dr/dtheta
            r_ = std::tanh(theta_);
            return;
        }
        W_ = kappa * W_ + 1.0;
        S_ = kappa * S_ + a * a + b * b;
        P_ = kappa * P_ + a * b;
        // f(-1) = S + 2P >= 0 and f(1) = 2P - S <= 0: a root in [-1, 1] always exists. Newton from the
        // previous root, kept inside a shrinking bracket (bisection when Newton leaves it).
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

    // Fresh sketches, and DataSketches' coin generator reseeded so the run is reproducible.
    void make_ds() {
#ifdef RECURSIVE_MPL_HAS_DS
        ds_margin::seed(c_.seed);
        dx_ = std::make_unique<ds_margin::Kll>(static_cast<std::uint16_t>(c_.kll_k));
        dy_ = std::make_unique<ds_margin::Kll>(static_cast<std::uint16_t>(c_.kll_k));
#endif
    }

    Config c_;
    const char* name_;
    std::size_t n_ = 0;
    std::size_t m_ = 0;   // copula updates so far
    double r_ = 0.0;
    double wsum_ = 0.0;   // accumulated EW weight, for the normalised start-up gain
    double gain_ = c_.gain;   // current gain (moved only by set_neff)
    double theta_ = 0.0, H_ = 1.0;          // FisherTheta
    double W_ = 0.0, S_ = 0.0, P_ = 0.0;    // Exact
    KllSketch kx_, ky_;
#ifdef RECURSIVE_MPL_HAS_DS
    std::unique_ptr<ds_margin::Kll> dx_, dy_;   // only for Margin::DsKll
#endif
    RankPit px_, py_;
    EwMarginal ex_{c_.margin == Margin::Ew ? c_.marginal_neff : 0.0}, ey_{c_.margin == Margin::Ew ? c_.marginal_neff : 0.0};
    copula::CopulaFit fit_;
};
