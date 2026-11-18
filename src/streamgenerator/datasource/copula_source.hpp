#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include "copula.hpp"
#include "dependence.hpp"

// Bivariate copula source with time-varying dependence. The Dependence supplies a
// scalar knob d per step (the Gaussian-equivalent linear correlation); the Copula
// family maps it to the SAME target Spearman rho_S = (6/pi) asin(d/2) and samples
// two standard-normal scores whose dependence follows that family (Gaussian /
// Clayton / Gumbel / Student-t -- see src/.../copula.hpp). The scores double as
// standardised "returns" fed to the estimators, and the true (achieved) Spearman
// rho is carried alongside as ground truth. With the default Gaussian copula this
// reproduces the original closed-form Gaussian construction exactly.
class CopulaSource {
public:
    struct Sample {
        double x;      // standardised return / normal score of series X
        double y;      // standardised return / normal score of series Y
        double rho;    // dependence knob d (Gaussian-equivalent linear correlation)
        double rho_s;  // true Spearman's rho at this step
    };

    CopulaSource(Dependence& dependence, dgp::Copula& copula, unsigned seed)
        : dependence_(dependence), copula_(copula), rng_(seed) {}

    Sample next() {
        const double d = std::clamp(dependence_.next(), -0.999, 0.999);
        const dgp::CopulaDraw draw = copula_.sample(d, rng_);
        return {draw.z1, draw.z2, d, draw.rho_s};
    }

private:
    Dependence& dependence_;
    dgp::Copula& copula_;
    std::mt19937 rng_;
};
