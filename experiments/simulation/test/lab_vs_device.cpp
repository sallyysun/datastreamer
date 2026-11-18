// The browser lab's MFF with the Gaussian working family and the thesis memories must equal the Chapter 6 device's
// MFF (and its VFF at rate 0.02) tick by tick. Usage: lab_vs_device <copula> <scenario> <seed>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../../../src/consumer/live_lab.hpp"
#include "../../../src/estimators/device.hpp"
#include "../../../src/streamgenerator/datasource/copula.hpp"
#include "../../../src/streamgenerator/datasource/dependence.hpp"
#include "../../../src/streamgenerator/datasource/copula.hpp"

int main(int argc, char** argv) {
    const std::string cop = argv[1], sc = argv[2];
    const unsigned seed = unsigned(std::atoi(argv[3]));
    const std::vector<double> mem(thesis::kMem, thesis::kMem + thesis::kK);
    for (const std::string est : {"rmpl", "xiao", "hermite"}) {
        // Separate passes: the KLL sketches share one global compaction-coin generator.
        const std::size_t e = est == "rmpl" ? 0 : est == "xiao" ? 1 : 2;
        std::vector<double> dm, dv, lm, lv, f;
        {
            auto dep = make_scenario(sc, thesis::kN); auto c = dgp::make_copula(cop); CopulaSource src(*dep, *c, seed);
            thesis::Device dev;
            for (std::size_t t = 0; t < thesis::kN; ++t) {
                dev.sets()[e].forecasts(f); dm.push_back(f[thesis::kK]); dv.push_back(f[thesis::kK + 1]);
                const auto s = src.next(); dev.update(s.x, s.y);
            }
        }
        {
            auto dep = make_scenario(sc, thesis::kN); auto c = dgp::make_copula(cop); CopulaSource src(*dep, *c, seed);
            datasketches_kll::seed(1);
            LiveLab lab(est, mem, {"gaussian"}, 1.0, 1e-4);
            for (std::size_t t = 0; t < thesis::kN; ++t) {
                lm.push_back(lab.mff()); lv.push_back(lab.vff());
                const auto s = src.next(); lab.update(s.x, s.y);
            }
        }
        double wm = 0, wv = 0;
        for (std::size_t t = 0; t < thesis::kN; ++t) { wm = std::max(wm, std::fabs(dm[t] - lm[t])); wv = std::max(wv, std::fabs(dv[t] - lv[t])); }
        std::printf("%s %s %s: max |lab MFF - device MFF| %g, max |lab VFF - device VFF| %g\n", cop.c_str(), sc.c_str(), est.c_str(), wm, wv);
    }
}
