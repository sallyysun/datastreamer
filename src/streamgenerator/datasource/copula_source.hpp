#pragma once

#include <algorithm>
#include <random>

#include "copula.hpp"
#include "dependence.hpp"

// Draw paired normal scores from a seeded copula with a time-varying dependence schedule.
// Return the Gaussian-equivalent correlation and achieved Spearman rho alongside each draw.
class CopulaSource {
public:
    CopulaSource(Dependence& dependence, dgp::Copula& copula, unsigned seed)
        : dependence_(dependence), copula_(copula), rng_(seed) {}

    dgp::CopulaDraw next() {
        return draw(dependence_, copula_, rng_);
    }

    static dgp::CopulaDraw draw(Dependence& dependence, dgp::Copula& sampler, std::mt19937& rng) {
        const double knob = std::clamp(dependence.next(), copula::Gaussian::kMinParam, copula::Gaussian::kMaxParam);
        return sampler.sample(knob, rng);
    }

private:
    Dependence& dependence_;
    dgp::Copula& copula_;
    std::mt19937 rng_;
};
