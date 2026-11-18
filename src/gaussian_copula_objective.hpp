#pragma once

#include <cmath>

// Bivariate Gaussian-copula objective for the parametric branch (online MLE /
// Fisher scoring). Works in normal scores z_x = Phi^{-1}(u), z_y = Phi^{-1}(v);
// for Gaussian marginals these are just the standardized observations.
//
// `rho` is the copula correlation. Derivatives are w.r.t. rho:
//   log_likelihood  l(z_x, z_y; rho)
//   score           s = dl/drho              (unbiased: E[s] = 0 at the true rho)
//   fisher          I(rho) = -E[d2l/drho2]
//                          = (1 + rho^2) / (1 - rho^2)^2   (closed form)
//
// Log-density of the Gaussian copula in normal scores:
//   l = -1/2 log(1-rho^2) - ( rho^2 (z_x^2 + z_y^2) - 2 rho z_x z_y ) / (2(1-rho^2)).
struct GaussianCopulaObjective {
    static double log_likelihood(double zx, double zy, double rho) {
        const double d = 1.0 - rho * rho;
        const double q = rho * rho * (zx * zx + zy * zy) - 2.0 * rho * zx * zy;
        return -0.5 * std::log(d) - q / (2.0 * d);
    }

    // dl/drho = [ rho(1-rho^2) - rho(z_x^2+z_y^2) + z_x z_y (1+rho^2) ] / (1-rho^2)^2.
    static double score(double zx, double zy, double rho) {
        const double d = 1.0 - rho * rho;
        return rho / d + ((1.0 + rho * rho) * zx * zy - rho * (zx * zx + zy * zy)) / (d * d);
    }

    static double fisher(double rho) {
        const double d = 1.0 - rho * rho;
        return (1.0 + rho * rho) / (d * d);
    }
};
