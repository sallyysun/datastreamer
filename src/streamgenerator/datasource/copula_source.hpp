#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include "copula.hpp"
#include "dependence.hpp"

// Draw paired normal scores from a seeded copula with a time-varying dependence schedule.
// Return the Gaussian-equivalent correlation and achieved Spearman rho alongside each draw.
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
