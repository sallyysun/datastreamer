// E2 constant-dependence figure (chapter6_central_results.md §2): the five fixed arms and the no-forgetting arm of
// <est>cum_bank for Xiao, Hermite and MPL, on the streams of probe_thesis_bank (same seeds, paths, shared margins).
// The first five arms are the E3 fixed arms; the probe's check compares them with thesis_bank_matched500[_fam].csv.
// Usage (from experiments/): SCENARIOS=matched ./probe_nocum_arms [seeds=60]   (COPULA=gaussian|clayton|gumbel|t)
//   -> results/nocum_arms[_<family>].csv   copula,scenario,seed,component,window,method,rmse

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
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
const std::vector<std::string> kArm{"fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "nocum"};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return scenario_drift(kN);
    return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
}
struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") { w.push_back({"pre", kBurn, kChange}); w.push_back({"post", kChange, kN}); }
    return w;
}
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    std::FILE* out = std::fopen(("results/nocum_arms" + (cop == "gaussian" ? std::string() : "_" + cop) + ".csv").c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,component,window,method,rmse\n");
    for (const std::string sc : {"constant", "drift", "jump"})
        for (int s = 0; s < seeds; ++s)
            for (const std::string c : {"xiao", "hermite", "mle"}) {
                auto dep = make_dep(sc);
                auto cp = dgp::make_copula(cop);
                CopulaSource src(*dep, *cp, 42u + s);
                auto e = make_estimator(c + "cum_bank");
                auto* b = dynamic_cast<ForgettingBank*>(e.get());
                const std::size_t K = b->size();
                std::vector<std::vector<double>> se(K, std::vector<double>(kN));
                for (std::size_t t = 0; t < kN; ++t) {
                    const auto smp = src.next();
                    for (std::size_t k = 0; k < K; ++k) {   // forecast held before tick t
                        const double d = b->model(k).estimate() - smp.rho_s;
                        se[k][t] = d * d;
                    }
                    b->update(smp.x, smp.y);
                }
                for (const auto& w : windows(sc))
                    for (std::size_t k = 0; k < K; ++k) {
                        double a = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) a += se[k][t];
                        std::fprintf(out, "%s,%s,%u,%s,%s,%s,%.6f\n", cop.c_str(), sc.c_str(), 42u + s, c.c_str(), w.name,
                                     kArm[k].c_str(), std::sqrt(a / double(w.b - w.a)));
                    }
            }
    std::fclose(out);
}
