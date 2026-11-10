// Mean bank weights through a jump, for plain Hedge (softmax, delta = 0), fixed share (delta = 3e-4,
// the thesis log bank) and fixed share with a cumulative arm. MPL arms (spec, start-up rule), log loss,
// eta = 1, the bank's shared 500-tick rank PIT. See notes/studies/budget_mpl.md §16.
// Usage (from experiments/): ./probe_bank_weights [seeds=60] (COPULA=..., SCENARIO=jump|drift)
//   -> results/bank_weights_paths[_<scenario>][_<fam>].csv   variant,t,k,neff,weight  and  variant,t,-1,0,sq_err
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "scenarios.hpp"

namespace {
constexpr std::size_t kN = 5000;
const std::vector<double> kGains{0.05, 0.02, 0.01, 0.005, 0.001};

std::unique_ptr<ForgettingBank> make_bank(double delta, bool cum) {
    std::vector<std::unique_ptr<DependenceEstimator>> m;
    for (double g : kGains) {
        RecursiveMPL::Config c{};
        c.margin = RecursiveMPL::Margin::ExactNormal; c.gain = g; c.startup = true;
        m.push_back(std::make_unique<RecursiveMPL>(c, "m"));
    }
    if (cum) {
        RecursiveMPL::Config c{};
        c.margin = RecursiveMPL::Margin::ExactNormal; c.gain = 1e-12; c.startup = true;
        m.push_back(std::make_unique<RecursiveMPL>(c, "m"));
    }
    ForgettingBank::Config cfg{.loss = ForgettingBank::Loss::Log, .eta = 1.0, .delta = delta};
    return std::make_unique<ForgettingBank>(std::move(m), cfg, "b");
}
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const std::string cop = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const std::string sc = std::getenv("SCENARIO") ? std::getenv("SCENARIO") : "jump";
    const std::vector<std::pair<std::string, std::pair<double, bool>>> variants{
        {"softmax", {0.0, false}}, {"fixed_share", {3e-4, false}}, {"fixed_share_cum", {3e-4, true}}};
    std::FILE* out = std::fopen(("results/bank_weights_paths" + (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string()) + std::string(sc == "jump" ? "" : "_" + sc) +
                                 std::string(cop == "gaussian" ? "" : "_" + cop) + ".csv").c_str(), "w");
    std::fprintf(out, "variant,t,k,neff,value\n");
    for (const auto& [name, cfg] : variants) {
        const std::size_t K = kGains.size() + (cfg.second ? 1 : 0);
        std::vector<std::vector<double>> w(kN, std::vector<double>(K, 0.0));
        std::vector<double> se(kN, 0.0);
        for (int s = 0; s < seeds; ++s) {
            std::unique_ptr<Dependence> dep;
            if (sc == "drift") dep = scenario_drift(kN);
            else dep = std::make_unique<JumpDependence>(0.2, 0.8, kN / 2);
            auto c = dgp::make_copula(cop);
            CopulaSource src(*dep, *c, 42 + s);
            auto b = make_bank(cfg.first, cfg.second);
            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                const double e = b->estimate() - smp.rho_s;
                se[t] += e * e;
                b->update(smp.x, smp.y);
                for (std::size_t k = 0; k < K; ++k) w[t][k] += b->weight(k);
            }
        }
        for (std::size_t t = 0; t < kN; t += 5) {
            for (std::size_t k = 0; k < K; ++k) {
                const double neff = k < kGains.size() ? (2.0 - kGains[k]) / kGains[k] : static_cast<double>(t + 1);
                std::fprintf(out, "%s,%zu,%zu,%.1f,%.6f\n", name.c_str(), t, k, neff, w[t][k] / seeds);
            }
            std::fprintf(out, "%s,%zu,-1,0,%.8f\n", name.c_str(), t, se[t] / seeds);
        }
    }
    std::fclose(out);
}
