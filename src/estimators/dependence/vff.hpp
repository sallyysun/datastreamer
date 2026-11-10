#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "estimator.hpp"
#include "ew_marginal.hpp"
#include "marginal_rank_pit.hpp"
#include "memory_control.hpp"

// Common variable forgetting factor (one rule for Xiao, Hermite and RMPL). A controller adapts the
// memory of the ideal estimator of thesis 5.4.4, the normalised EW mean m_n of
// p_n = 12 (U_n - 1/2)(V_n - 1/2) built from the shared pseudo-observations, by stochastic gradient
// descent on its prequential squared loss (p_n - m_{n-1})^2 (Bodenham & Adams 2017's adaptive
// forgetting factor), with the derivative of m with respect to lambda carried recursively. The step is
// taken in theta = log N_eff, scaled by a running RMS of the gradient, and N_eff is clamped to the
// bank's span. Each wrapped estimator's memory is set to the controller's N_eff before it updates.
// Design and validation: notes/studies/ew_marginals.md §12.
class VffController {
public:
    struct Config {
        double rate = 0.05;           // step in log N_eff per unit normalised gradient
        double neff_min = 39.0, neff_max = 1999.0;
        double neff_start = 199.0;
        double rms_rate = 0.01;       // smoothing of the gradient's squared scale
    };

    explicit VffController(Config c) : c_(c), theta_(std::log(c.neff_start)) {}

    // Prequential: the loss of the mean held through n-1 against p_n, then the memory step, then the
    // mean absorbs p_n under the new memory.
    void observe(double p) {
        if (w_ > 0.0) {
            const double e = p - m_;
            const double n = neff();
            const double dlam_dtheta = 2.0 * n / ((n + 1.0) * (n + 1.0));
            const double g = -2.0 * e * dm_ * dlam_dtheta;     // d (p - m)^2 / d theta
            v_ += c_.rms_rate * (g * g - v_);
            if (v_ > 0.0) theta_ -= c_.rate * g / std::sqrt(v_);
            theta_ = std::clamp(theta_, std::log(c_.neff_min), std::log(c_.neff_max));
        }
        const double n = neff(), lam = (n - 1.0) / (n + 1.0);
        const double w_old = w_;
        w_ = lam * w_ + 1.0;
        dw_ = lam * dw_ + w_old;                               // d w / d lambda
        const double e = p - m_;
        dm_ = dm_ * (1.0 - 1.0 / w_) - e * dw_ / (w_ * w_);    // d m / d lambda
        m_ += e / w_;
    }

    double neff() const { return std::exp(theta_); }
    void reset() { theta_ = std::log(c_.neff_start); m_ = w_ = dm_ = dw_ = v_ = 0.0; }

private:
    Config c_;
    double theta_;
    double m_ = 0.0, w_ = 0.0, dm_ = 0.0, dw_ = 0.0, v_ = 0.0;
};

// A bank-protocol estimator with the common VFF: the shared EW margins give (U_n, V_n) before the update,
// the controller adapts, the estimator's memory is set, then the estimator updates on raw (X, Y) (Xiao,
// Hermite) or on the normal scores (RMPL), exactly as a bank arm would.
class CommonVff : public DependenceEstimator {
public:
    CommonVff(std::unique_ptr<DependenceEstimator> inner, VffController::Config c, bool raw_inputs, const char* name)
        : inner_(std::move(inner)), mc_(dynamic_cast<MemoryControlled*>(inner_.get())), ctl_(c),
          raw_(raw_inputs), name_(name) {}

    void update(double x, double y) override {
        const double u = mx_.transform(x), v = my_.transform(y);
        if (mx_.size() > 30) ctl_.observe(12.0 * (u - 0.5) * (v - 0.5));
        if (mc_) mc_->set_neff(ctl_.neff());
        if (raw_) inner_->update(x, y);
        else inner_->update(RankPit::probit(u), RankPit::probit(v));
    }

    double estimate() const override { return inner_->estimate(); }
    bool ready() const override { return inner_->ready(); }
    void reset() override { inner_->reset(); ctl_.reset(); mx_.reset(); my_.reset(); }
    std::size_t store_bytes() const override {
        return sizeof(*this) + inner_->store_bytes() + mx_.heap_bytes() + my_.heap_bytes();
    }
    const char* name() const override { return name_; }
    double neff() const { return ctl_.neff(); }

private:
    std::unique_ptr<DependenceEstimator> inner_;
    MemoryControlled* mc_;
    VffController ctl_;
    EwMarginal mx_{default_marginal_neff()}, my_{default_marginal_neff()};
    bool raw_;
    const char* name_;
};
