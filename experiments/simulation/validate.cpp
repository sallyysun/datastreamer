// VALIDATE=1: compare the tabulated densities and Hellinger distances with direct calculations.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "strategy_runs.hpp"
#include "working_families.hpp"

namespace strategy_experiments {

using thesis::gauss_logc;
using thesis::H2Table;

namespace {

double density(copula::Family family, double u, double v, double parameter) {
    const double log_density = copula::log_density(family, u, v, parameter);
    return std::isfinite(log_density) ? std::exp(log_density) : 0.0;
}

// The working-family tables used by SCORESFAM and FAMSWITCH.
void validate_family_tables(H2Table& h2) {
    const WorkingFamilies families;
    const auto& family = WorkingFamilies::family;
    double inverse_error = 0;
    for (std::size_t wi = 1; wi < 4; ++wi)
        for (double r = -0.2; r < 0.995; r += 0.0137) {
            const double a = families.inverse[wi](r), b = copula::param_from_spearman(family[wi], r);
            inverse_error = std::max(inverse_error, std::fabs(a - b) / std::max(1e-3, std::fabs(b)));
        }
    std::printf("VALIDATE inverse table max relative error %.2e\n", inverse_error);

    FamilyHellinger hellinger(h2, families.inverse, true);
    {   // one working family per mixture, against the run's generating family
        const double par = copula::param_from_spearman(h2.F, 0.55);
        const double ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15};
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (const auto& rs : std::vector<std::array<double, 5>>{{0.21, 0.4567, 0.6, 0.78, 0.91}, {0.05, 0.33, 0.52, 0.5501, 0.97}, {-0.2, 0.1, 0.3, 0.7, 0.85}}) {
                double th[5];
                for (int k = 0; k < 5; ++k) th[k] = families.inverse[wi](rs[k]);
                const dmetric::Table truth(h2.g, [&](double a, double b) { return density(h2.F, a, b, par); });
                const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * density(family[wi], a, b, th[k]); return m; });
                const double exact = dmetric::hellinger2(h2.g, truth, mt);
                worst = std::max(worst, std::fabs(hellinger.single_working_family(wi, par, rs.data(), ws, 5) - exact) / exact);
            }
        std::printf("VALIDATE working-family mixture H2 max relative error %.2e\n", worst);
    }
    {   // mixed working families, one mixture per non-Gaussian truth
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi) {
            const int wis[3] = {wi, wi, 0};
            const double rhs[3] = {0.31, 0.6123, 0.55}, wss[3] = {0.3, 0.5, 0.2};
            const double par = copula::param_from_spearman(family[wi], thesis::kConstantRhoS);
            const dmetric::Table tt(h2.g, [&](double a, double b) { return density(family[wi], a, b, par); });
            const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 3; ++k) m += wss[k] * density(family[wis[k]], a, b, copula::param_from_spearman(family[wis[k]], rhs[k])); return m; });
            const double exact = dmetric::hellinger2(h2.g, tt, mt);
            worst = std::max(worst, std::fabs(hellinger.mixed_families(family[wi], par, wis, rhs, wss, 3) - exact) / exact);
        }
        std::printf("VALIDATE mixed-family mixture H2 max relative error %.2e\n", worst);
    }
}

}  // namespace

int run_validate(const Run& run) {
    const std::string& cop = run.cop;
    H2Table& h2 = run.h2;
    const double par = copula::param_from_spearman(h2.F, 0.6 * 0.97);
    for (double r : {0.123, 0.4567, 0.58, 0.7891, 0.9})
        std::printf("VALIDATE H2 %s rho_hat %.4f: table %.6e direct %.6e\n", cop.c_str(), r, h2(par, r), h2.direct(par, r));
    {   // mixture H^2: closed-form grid vs a dmetric::Table of the same mixture through copula::log_density
        const double rs[5] = {0.2, 0.45, 0.6, 0.75, 0.9}, ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15}, w1[5] = {0, 0, 1, 0, 0};
        const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], a, b)); return m; });
        std::printf("VALIDATE H2mix %s 5-comp: grid %.6e dmetric %.6e | 1-comp (rho 0.6): grid %.6e direct %.6e\n", cop.c_str(),
                    h2.mix(par, rs, ws, 5), dmetric::hellinger2(h2.g, truth, mt), h2.mix(par, rs, w1, 5), h2.direct(par, 0.6));
        const double lm = H2Table::mix_logc(rs, ws, 5, 0.3, 0.8);
        double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], 0.3, 0.8));
        std::printf("VALIDATE mix_logc: %.10f vs %.10f\n", lm, std::log(m));
    }
    validate_family_tables(h2);
    return 0;
}

}  // namespace strategy_experiments
