// Validation of the common VFF rule (vff.hpp) on development seeds: memory paths and RMSE of the
// VFF-wrapped Xiao, Hermite and RMPL for several controller rates. notes/studies/ew_marginals.md §12.
// Usage (from experiments/): SEED_BASE=1 ./probe_vff_common [seeds=30]   (COPULA=...)
//   -> results/vff_common[_fam].csv (scenario,rate,estimator,seed,rmse) and vff_common_paths[_fam].csv
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/vff.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "scenarios.hpp"

namespace {
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return scenario_drift(kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return scenario_oscillate();
}
std::unique_ptr<CommonVff> make(const std::string& est, double rate) {
    VffController::Config c{.rate = rate};
    if (est == "xiao")
        return std::make_unique<CommonVff>(std::make_unique<XiaoSpearmanEstimator>(bank_experiment::xiao(0.99), "x"), c, true, "vff");
    if (est == "hermite")
        return std::make_unique<CommonVff>(std::make_unique<HermiteEstimator>(bank_experiment::hermite(0.01), "h"), c, true, "vff");
    RecursiveMPL::Config m{};
    m.margin = RecursiveMPL::Margin::ExactNormal;
    m.gain = 0.01;
    bank_experiment::rmpl(m);
    return std::make_unique<CommonVff>(std::make_unique<RecursiveMPL>(m, "m"), c, false, "vff");
}
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 30;
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : 1u;
    const std::string fam = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const std::string sfx = (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string()) + (fam == "gaussian" ? "" : "_" + fam);
    const char* rates_env = std::getenv("VFF_RATES");
    std::vector<double> rates{0.005, 0.02, 0.05, 0.1};
    if (rates_env) { rates.clear(); for (const char* p = rates_env; *p;) { rates.push_back(std::strtod(p, const_cast<char**>(&p))); if (*p == ',') ++p; } }
    std::FILE* out = std::fopen(("results/vff_common" + sfx + ".csv").c_str(), "w");
    std::FILE* pout = std::fopen(("results/vff_common_paths" + sfx + ".csv").c_str(), "w");
    std::fprintf(out, "copula,scenario,rate,estimator,seed,window,rmse\n");
    std::fprintf(pout, "copula,scenario,rate,t,neff\n");
    for (const std::string sc : {"constant", "drift", "jump", "oscillate"})
        for (double rate : rates) {
            std::vector<double> path(kN / 25, 0.0);
            for (const std::string est : {"xiao", "hermite", "mle"})
                for (int s = 0; s < seeds; ++s) {
                    auto dep = make_dep(sc);
                    auto cop = dgp::make_copula(fam);
                    CopulaSource src(*dep, *cop, base + s);
                    auto e = make(est, rate);
                    double se = 0, sp = 0, so = 0; std::size_t n = 0, np = 0, no = 0;
                    for (std::size_t t = 0; t < kN; ++t) {
                        const auto smp = src.next();
                        const double r = e->estimate(), err = (r - smp.rho_s) * (r - smp.rho_s);
                        if (t >= kBurn) { se += err; ++n; }
                        if (t >= kChange && t < kChange + 500) { sp += err; ++np; }
                        if (t >= kBurn && t < kChange) { so += err; ++no; }
                        e->update(smp.x, smp.y);
                        if (est == "xiao" && t % 25 == 0) path[t / 25] += e->neff() / seeds;
                    }
                    std::fprintf(out, "%s,%s,%g,%s,%u,whole,%.6f\n", fam.c_str(), sc.c_str(), rate, est.c_str(), base + s, std::sqrt(se / n));
                    std::fprintf(out, "%s,%s,%g,%s,%u,post500,%.6f\n", fam.c_str(), sc.c_str(), rate, est.c_str(), base + s, std::sqrt(sp / np));
                    std::fprintf(out, "%s,%s,%g,%s,%u,pre,%.6f\n", fam.c_str(), sc.c_str(), rate, est.c_str(), base + s, std::sqrt(so / no));
                }
            for (std::size_t i = 0; i < path.size(); ++i)
                std::fprintf(pout, "%s,%s,%g,%zu,%.2f\n", fam.c_str(), sc.c_str(), rate, i * 25, path[i]);
        }
    std::fclose(out); std::fclose(pout);
    return 0;
}
