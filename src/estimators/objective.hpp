#pragma once

#include <cmath>

// Bivariate Gaussian-copula log-likelihood, score and Fisher information in rho, on normal scores
// z = Phi^{-1}(u). Formulas: notes/design/source_rationale.md.
struct GaussianCopulaObjective {
    static double log_likelihood(double zx, double zy, double rho) {
        const double d = 1.0 - rho * rho;
        const double q = rho * rho * (zx * zx + zy * zy) - 2.0 * rho * zx * zy;
        return -0.5 * std::log(d) - q / (2.0 * d);
    }

    static double score(double zx, double zy, double rho) {
        const double d = 1.0 - rho * rho;
        const double a = zx * zx + zy * zy;
        const double b = zx * zy;
        return (rho * d - rho * a + b * (1.0 + rho * rho)) / (d * d);
    }

    static double fisher(double rho) {
        const double d = 1.0 - rho * rho;
        return (1.0 + rho * rho) / (d * d);
    }
};
