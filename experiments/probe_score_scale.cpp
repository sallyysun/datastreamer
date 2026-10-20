// Is score / sqrt(i_theta) still variance 1 when the copula family is wrong?
//
// Under correct specification the information matrix equality gives
// Var[s] = I, so the standardized score is the unit-variance innovation a CUSUM
// assumes. Under misspecification the equality fails (White 1982) and the ratio
// Var[s]/I is exactly the factor by which every calibrated threshold is wrong.
// See section 14.12 of notes/studies/kalman_changepoint_results.md.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/gaussian_mle.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main() {
    const std::vector<std::string> families = {"gaussian", "clayton", "gumbel", "t"};
    const std::vector<double> rhos = {0.3, 0.6, 0.8};
    const std::size_t N = 200000, burn = 20000, block = 20;

    std::printf("%-10s", "family");
    for (double r : rhos) std::printf("  rho=%.1f: Var[s~]  sd-ratio", r);
    std::printf("\n");

    const double robust = (std::getenv("ROBUST") != nullptr) ? std::atof(std::getenv("ROBUST")) : 0.0;
    for (const std::string& fam : families) {
        std::printf("%-10s", fam.c_str());
        for (double rho : rhos) {
            ConstantDependence dep(rho);
            auto cop = dgp::make_copula(fam);
            CopulaSource src(dep, *cop, 7);
            GaussianCopulaMLE mle(GaussianCopulaMLE::Config{.lambda = 0.99, .burn_in = 50, .robust_score_rate = robust}, "p");

            // Variance of the BLOCK statistic, which is what the CUSUM sees.
            double sum = 0.0, s1 = 0.0, s2 = 0.0;
            std::size_t in_block = 0, n = 0;
            for (std::size_t t = 0; t < N; ++t) {
                const CopulaSource::Sample s = src.next();
                mle.update(s.x, s.y);
                if (t < burn) continue;
                sum += mle.standardized_score();
                if (++in_block < block) continue;
                const double e = sum / std::sqrt(static_cast<double>(block));
                sum = 0.0; in_block = 0;
                s1 += e; s2 += e * e; ++n;
            }
            const double mean = s1 / static_cast<double>(n);
            const double var = s2 / static_cast<double>(n) - mean * mean;
            std::printf("  %14.3f  %8.2f", var, std::sqrt(var));
        }
        std::printf("\n");
    }
    std::printf("\nsd-ratio is the factor every ARL0-calibrated threshold is off by.\n");
    return 0;
}
