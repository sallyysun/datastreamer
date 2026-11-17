#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

#include "../registry.hpp"
#include "../special_functions.hpp"


// Common variable forgetting factor (one rule for Xiao, Hermite and RMPL). A controller adapts the
// memory of the ideal estimator of thesis 5.4.4, the normalised EW mean m_n of
// p_n = 12 (U_n - 1/2)(V_n - 1/2) built from the shared pseudo-observations, by stochastic gradient
// descent on its prequential squared loss (p_n - m_{n-1})^2 (Bodenham & Adams 2017's adaptive
// forgetting factor), with the derivative of m with respect to lambda carried recursively. The step is
// taken in theta = log N_eff, scaled by a running RMS of the gradient, and N_eff is clamped to the
// bank's span. Each wrapped estimator's memory is set to the controller's N_eff before it updates.
// Used by the VFF strategy (evaluation.hpp). Design and validation: archive/notes/studies/ew_marginals.md §12.
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

namespace thesis {

// VFF: one estimator whose N_eff follows VffController, fed p_n = 12 (U - 1/2)(V - 1/2) once 30 margin observations
// are held; Counting Matrix / Hermite read the raw observations, RMPL reads probit(U).
struct Vff {
    std::unique_ptr<DependenceEstimator> inner;
    VffController ctl;
    Vff(const std::string& e, double rate) : inner(arm(e, 199, "m")),
                                             ctl(VffController::Config{.rate = rate}) {}
    // n_after: margin observations held after absorbing this tick
    void update(double x, double y, double u, double v, std::size_t n_after, bool raw) {
        if (n_after > 30) ctl.observe(12.0 * (u - 0.5) * (v - 0.5));
        inner->set_neff(ctl.neff());
        if (raw) inner->update(x, y); else inner->update(special::probit(u), special::probit(v));
    }
};

}  // namespace thesis
