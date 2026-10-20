// Does the tail/middle ratio separate copula SHAPE while ignoring rho?
// If it moves with rho it is not a shape statistic and the detector is pointless.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../src/estimators/dependence/tail_cusum.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main() {
    const std::vector<std::string> fams = {"gaussian", "t", "clayton", "frechet"};
    const std::vector<double> ds = {0.4, 0.6, 0.8};
    const double q = 0.15;
    const std::size_t N = 400000, burn = 2000;

    std::printf("%-10s", "family");
    for (double d : ds) std::printf("   d=%.1f:  rho_S       r  excess", d);
    std::printf("\n");
    for (const std::string& f : fams) {
        std::printf("%-10s", f.c_str());
        for (double d : ds) {
            ConstantDependence dep(d);
            auto cop = dgp::make_copula(f);
            CopulaSource src(dep, *cop, 9);
            RankPit px(500), py(500);
            double a = 0, b = 0, rho_s = 0;
            std::size_t n = 0;
            for (std::size_t t = 0; t < N; ++t) {
                const auto s = src.next();
                const double u = px.transform(s.x), v = py.transform(s.y);
                if (px.size() < 250 || t < burn) continue;
                a += (u <= q && v <= q) ? 1.0 : 0.0;
                b += (u <= 0.5 && v <= 0.5) ? 1.0 : 0.0;
                rho_s += s.rho_s; ++n;
            }
            const double r = (a / n) / (b / n);
            // Excess over a Gaussian copula at the same rho -- the shape signal.
            TailCusumDetector probe(make_estimator("xiao_ew"), TailCusumDetector::Config{}, "p");
            const double base = probe.gaussian_ratio_for(rho_s / n);
            std::printf("   %7.3f %7.4f %+7.4f", rho_s / n, r, r - base);
        }
        std::printf("\n");
    }
    std::printf("\nr should be FLAT down each family's row (rho-invariant)\n");
    std::printf("and DIFFER between rows at the same d (shape-sensitive).\n");
    return 0;
}
