// thesis::Device + Scorer fed the probe's in-process stream: must reproduce scenarios MIXDENS=1 rows.
// Usage: device_vs_probe <copula> <scenario> <seed> [returns]  -> rows on stdout
// returns: feed the server's log returns ln(p_t / p_{t-1}), p_t = p_{t-1} exp(1e-4 + 0.01 z), instead of the draws z.
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../../../src/estimators/thesis_device.hpp"
#include "../../../src/streamgenerator/datasource/copula.hpp"
#include "../../../src/streamgenerator/datasource/dependence.hpp"
#include "../../../src/streamgenerator/datasource/copula_source.hpp"

int main(int argc, char** argv) {
    const std::string cop = argv[1], sc = argv[2];
    const unsigned seed = unsigned(std::atoi(argv[3]));
    auto dep = make_scenario(sc, thesis::kN);
    auto c = dgp::make_copula(cop);
    CopulaSource src(*dep, *c, seed);
    const bool ret = argc > 4;
    double px = 100, py = 100;
    thesis::Device dev;
    thesis::Scorer sco(cop, sc);
    const copula::Family F = thesis::family_of(cop);
    for (std::size_t t = 0; t < thesis::kN; ++t) {
        const auto s = src.next();
        const double par = copula::param_from_spearman(F, s.target);
        sco.observe(dev, t, s.rho_s, special::normal_cdf(s.x), special::normal_cdf(s.y), par);
        if (!ret) { dev.update(s.x, s.y); continue; }
        const double nx = px * std::exp(1e-4 + 0.01 * s.x), ny = py * std::exp(1e-4 + 0.01 * s.y);
        dev.update(std::log(nx / px), std::log(ny / py));
        px = nx; py = ny;
    }
    sco.write(stdout, dev, seed);
}
