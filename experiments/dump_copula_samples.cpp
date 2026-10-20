// Draws n points from each DGP copula family at one target Spearman rho, using
// the same samplers the harness uses, so the Chapter 3 figures show exactly the
// families the experiments run on. Output feeds plot_ch3.py.
//
// Usage: ./dump_copula_samples [rho_s] [n] [seed]   -> results/ch3_copula_samples.csv

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numbers>

#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main(int argc, char** argv) {
    const double rho_s = (argc > 1) ? std::atof(argv[1]) : 0.6;
    const int n = (argc > 2) ? std::atoi(argv[2]) : 2000;
    const unsigned seed = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42;

    // The dependence knob d is the Gaussian-equivalent correlation with this rho_S.
    const double d = 2.0 * std::sin(std::numbers::pi * rho_s / 6.0);

    std::ofstream out("results/ch3_copula_samples.csv");
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot write results/ (run: mkdir -p results)\n");
        return 1;
    }
    out << "family,param,rho_s_true,z1,z2\n";
    for (const char* name : {"gaussian", "t", "clayton", "gumbel"}) {
        auto cop = dgp::make_copula(name);
        ConstantDependence dep(d);
        CopulaSource src(dep, *cop, seed);
        const auto family = std::string(name) == "t" ? copula::Family::StudentT
                          : std::string(name) == "clayton" ? copula::Family::Clayton
                          : std::string(name) == "gumbel" ? copula::Family::Gumbel
                          : copula::Family::Gaussian;
        const double param = (family == copula::Family::Gaussian)
                                 ? d : dgp::param_from_spearman(family, rho_s);
        for (int i = 0; i < n; ++i) {
            const CopulaSource::Sample s = src.next();
            out << cop->name() << "," << param << "," << s.rho_s << "," << s.x << "," << s.y << "\n";
        }
        std::printf("%-9s param=%.4f\n", cop->name(), param);
    }
    return 0;
}
