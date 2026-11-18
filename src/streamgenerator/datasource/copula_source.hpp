#pragma once

#include <algorithm>
#include <random>

#include "copula.hpp"
#include "dependence.hpp"

// Draw paired normal scores from a seeded copula whose Spearman rho follows a dependence schedule.
class CopulaSource {
public:
    CopulaSource(Dependence& dependence, dgp::Copula& copula, unsigned seed)
        : dependence_(dependence), copula_(copula), rng_(seed) {}

    dgp::CopulaDraw next() {
        return draw(dependence_, copula_, rng_);
    }

    static dgp::CopulaDraw draw(Dependence& dependence, dgp::Copula& sampler, std::mt19937& rng) {
        return sampler.sample(std::clamp(dependence.next(), -1.0, 1.0), rng);
    }

private:
    Dependence& dependence_;
    dgp::Copula& copula_;
    std::mt19937 rng_;
};
