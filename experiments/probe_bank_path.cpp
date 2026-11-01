// Single-seed path of the log-loss Xiao bank (thesis Algorithm 1) on drift and jump: true rho_S,
// each fixed arm, the bank's rho and its weights, per tick. Illustration for section 6.3 only;
// the multi-seed numbers are probe_thesis_bank. Plot: plot_bank_path.py.
//
// Usage (from experiments/):  ./probe_bank_path [seed=42] [component=xiao]   (COPULA=gaussian|clayton|gumbel|t)
//   -> results/bank_path[_<family>].csv   scenario,t,truth,bank,arm_<M>...,w_<M>...

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main(int argc, char** argv) {
    constexpr std::size_t kN = 5000, kChange = 2500;
    const unsigned seed = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 42;
    const std::string comp = argc > 2 ? argv[2] : "xiao";
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const std::string sfx = cop_name == "gaussian" ? "" : "_" + cop_name;

    std::FILE* out = std::fopen(("results/bank_path" + sfx + ".csv").c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    bool header = false;
    for (const std::string sc : {"drift", "jump"}) {
        std::unique_ptr<Dependence> dep;
        if (sc == "drift") dep = std::make_unique<DriftDependence>(-0.5, 0.8, kN);
        else dep = std::make_unique<JumpDependence>(0.2, 0.8, kChange);
        auto cop = dgp::make_copula(cop_name);
        CopulaSource src(*dep, *cop, seed);
        auto e = make_estimator(comp + "_bank_log");
        auto* bank = dynamic_cast<ForgettingBank*>(e.get());
        const std::size_t K = bank->size();
        if (!header) {
            std::fprintf(out, "scenario,t,truth,bank");
            for (std::size_t k = 0; k < K; ++k) std::fprintf(out, ",arm_%zu", k);
            for (std::size_t k = 0; k < K; ++k) std::fprintf(out, ",w_%zu", k);
            std::fprintf(out, "\n");
            header = true;
        }
        for (std::size_t t = 0; t < kN; ++t) {
            const auto smp = src.next();
            std::fprintf(out, "%s,%zu,%.5f,%.5f", sc.c_str(), t, smp.rho_s, bank->estimate());
            for (std::size_t k = 0; k < K; ++k) std::fprintf(out, ",%.5f", bank->model(k).estimate());
            for (std::size_t k = 0; k < K; ++k) std::fprintf(out, ",%.5g", bank->weight(k));
            std::fprintf(out, "\n");
            bank->update(smp.x, smp.y);
        }
    }
    std::fclose(out);
    return 0;
}
