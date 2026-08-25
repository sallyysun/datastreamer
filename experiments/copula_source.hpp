#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

#include "../src/streamgenerator/datasource/dependence.hpp"

// Bivariate Gaussian-copula source with time-varying dependence. Each draw uses
// the current rho_t from the Dependence to correlate two standard normals, which
// double as standardised "returns" fed to the estimators. Because the DGP is a
// Gaussian copula, the true Spearman rho is known in closed form
// (rho_S = (6/pi) * asin(rho/2)), giving estimators a ground truth to track.
class CopulaSource {
public:
    struct Sample {
        double x;      // standardised return of series X
        double y;      // standardised return of series Y
        double rho;    // true copula (linear) correlation at this step
        double rho_s;  // true Spearman's rho at this step
    };

    CopulaSource(Dependence& dependence, unsigned seed)
        : dependence_(dependence), rng_(seed), normal_(0.0, 1.0) {}

    Sample next() {
        const double rho = std::clamp(dependence_.next(), -0.999, 0.999);
        const double z1 = normal_(rng_);
        const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng_);
        const double rho_s = 6.0 / std::numbers::pi * std::asin(rho / 2.0);
        return {z1, z2, rho, rho_s};
    }

private:
    Dependence& dependence_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};
