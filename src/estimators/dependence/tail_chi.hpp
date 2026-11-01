#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "estimator.hpp"
#include "marginal_rank_pit.hpp"
#include "tail_estimator.hpp"

// Model-free EW corner counter on a rank PIT: for each tracked q, chi_hat(q) = b / a with
// a = EW P(U <= q), b = EW P(U <= q, V <= q) (and the upper-corner analogues), update weight
// alpha = 2/(M+1). estimate() = 12 EW mean(U V) - 3, a rho_S. lambda is NaN: no model for the limit.
// The counter logic is TailCusumDetector's, without the chart. notes/studies/thesis_bank_tails.md.
class TailChi : public DependenceEstimator, public TailEstimator {
public:
    struct Config {
        double memory = 1999;
        std::array<double, 2> qs{0.05, 0.10};
        std::size_t pit_window = 500;
        std::size_t warm_up = 30;
    };

    TailChi(Config config, const char* name)
        : config_(config), name_(name), alpha_(2.0 / (config.memory + 1.0)),
          pit_x_(config.pit_window), pit_y_(config.pit_window) {}

    void update(double x, double y) override {
        const bool scorable = pit_x_.size() >= config_.warm_up;
        const double u = pit_x_.transform(x), v = pit_y_.transform(y);
        if (scorable) update_uv(u, v);
    }

    // The counters on given pseudo-observations (the probe's known-margin check feeds true PITs).
    void update_uv(double u, double v) {
        const double a = alpha_;
        uv_ += a * (u * v - uv_);
        for (std::size_t i = 0; i < config_.qs.size(); ++i) {
            const double q = config_.qs[i];
            const bool lu = u <= q, uu = u > 1.0 - q;
            aL_[i] += a * (lu - aL_[i]);
            bL_[i] += a * ((lu && v <= q) - bL_[i]);
            aU_[i] += a * (uu - aU_[i]);
            bU_[i] += a * ((uu && v > 1.0 - q) - bU_[i]);
        }
        ++n_;
    }

    double estimate() const override { return n_ ? 12.0 * uv_ - 3.0 : 0.0; }

    TailEstimate tails(double q) const override {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t i = 0; i < config_.qs.size(); ++i)
            if (std::fabs(q - config_.qs[i]) < 1e-12) {
                // Guard: before any marginal-corner event the ratio is undefined; report 0.
                const double cl = aL_[i] > 1e-12 ? bL_[i] / aL_[i] : 0.0;
                const double cu = aU_[i] > 1e-12 ? bU_[i] / aU_[i] : 0.0;
                return {cl, cu, kNaN, kNaN};
            }
        return {kNaN, kNaN, kNaN, kNaN};
    }

    void reset() override {
        pit_x_.reset();
        pit_y_.reset();
        uv_ = 0.25;
        aL_.fill(0.0); bL_.fill(0.0); aU_.fill(0.0); bU_.fill(0.0);
        n_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + pit_x_.store_bytes() + pit_y_.store_bytes() - 2 * sizeof(RankPit);
    }
    const char* name() const override { return name_; }

private:
    Config config_;
    const char* name_;
    double alpha_;
    RankPit pit_x_, pit_y_;
    double uv_ = 0.25;   // independence start
    std::array<double, 2> aL_{}, bL_{}, aU_{}, bU_{};
    std::size_t n_ = 0;
};
