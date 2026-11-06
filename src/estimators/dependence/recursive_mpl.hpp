#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "estimator.hpp"
#include "predictive_density.hpp"
#if __has_include(<kll_sketch.hpp>)   // DataSketches on the include path (experiments/Makefile DS_INC)
#include "ds_margin.hpp"
#define RECURSIVE_MPL_HAS_DS 1
#endif
#include "kll_sketch.hpp"
#include "marginal_rank_pit.hpp"
#include "parametric_copula.hpp"

// Recursive Fisher-scaled pseudo-likelihood for the Gaussian copula, with a constant gain:
//   r_n = clip(r_{n-1} + gain * I(r_{n-1})^{-1} s_n(r_{n-1}), +-r_max).
// Pseudo-observations come from the marginal state BEFORE the current pair is inserted.
// Spec and results: notes/studies/budget_mpl.md.
class RecursiveMPL : public DependenceEstimator, public PredictiveDensity {
public:
    enum class Margin { Kll, DsKll, ExactNormal, RankWindow };
    enum class Update { ConstantGain, CopulaFit };   // CopulaFit: the registered MPL's recursion

    struct Config {
        Margin margin = Margin::Kll;
        Update update = Update::ConstantGain;
        double gain = 0.01;
        std::size_t kll_k = 200;
        std::size_t window = 500;
        double delta = 1e-3;
        double r_max = 0.999;
        std::size_t warmup = 30;   // marginal observations before the copula updates
        // Gain max(gain, 1/m) on the m-th copula update: a running mean until m = 1/gain, then the
        // constant gain. Same rule as HermiteEstimator::init_first_obs. Off = constant from the start.
        bool startup = false;
        unsigned seed = 1;
    };

    RecursiveMPL(Config c, const char* name)
        : c_(c), name_(name), kx_(c.kll_k, 2 * c.seed + 1), ky_(c.kll_k, 2 * c.seed + 2),
          px_(c.window), py_(c.window),
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
        }
        ++n_;
        if (seen < c_.warmup) return;
        u = std::clamp(u, c_.delta, 1.0 - c_.delta);
        v = std::clamp(v, c_.delta, 1.0 - c_.delta);
        if (c_.update == Update::CopulaFit) { fit_.update_uv(u, v); return; }
        const double z1 = RankPit::probit(u), z2 = RankPit::probit(v);
        const double r = r_, q = 1.0 - r * r;
        const double s = r / q + ((1.0 + r * r) * z1 * z2 - r * (z1 * z1 + z2 * z2)) / (q * q);
        const double info = (1.0 + r * r) / (q * q);
        ++m_;
        const double g = c_.startup ? std::max(c_.gain, 1.0 / static_cast<double>(m_)) : c_.gain;
        r_ = std::clamp(r + g * s / info, -c_.r_max, c_.r_max);
    }

    double estimate() const override {
        if (c_.update == Update::CopulaFit) return fit_.spearman();
        return (6.0 / M_PI) * std::asin(r_ / 2.0);
    }

    void reset() override {
        n_ = 0; m_ = 0; r_ = 0.0;
        kx_.reset(); ky_.reset(); px_.reset(); py_.reset(); fit_.reset();
        if (c_.margin == Margin::DsKll) make_ds();
    }

    // Marginal state plus r (or CopulaFit's phi and information); the unused margin is not counted.
    std::size_t store_bytes() const override {
        const std::size_t core = (c_.update == Update::CopulaFit ? 2 : 1) * sizeof(double) + sizeof(c_);
        switch (c_.margin) {
            case Margin::Kll:        return core + kx_.store_bytes() + ky_.store_bytes();
#ifdef RECURSIVE_MPL_HAS_DS
            case Margin::DsKll:      return core + dx_->store_bytes() + dy_->store_bytes();
#endif
            case Margin::RankWindow: return core + px_.store_bytes() + py_.store_bytes();
            default:                 return core;
        }
    }

    const char* name() const override { return name_; }

private:
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
    KllSketch kx_, ky_;
#ifdef RECURSIVE_MPL_HAS_DS
    std::unique_ptr<ds_margin::Kll> dx_, dy_;   // only for Margin::DsKll
#endif
    RankPit px_, py_;
    copula::CopulaFit fit_;
};
