// Single-seed path of the MPL log bank on drift and jump for two weighting rules: softmax only
// (plain Hedge, delta = 0) and softmax + fixed share (delta = 3e-4, the thesis bank). Per tick: truth,
// bank, each arm, each weight. Illustration only (multi-seed: probe_bank_weights). budget_mpl.md §16.
// Usage (from experiments/): ./probe_bank_path_variants [seed=42]  (COPULA=...)
//   -> results/bank_path_variants[_<fam>].csv   variant,scenario,t,truth,bank,arm_k...,w_k...
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 42;
    const std::string cop = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const std::vector<double> gains{0.05, 0.02, 0.01, 0.005, 0.001};
    std::FILE* out = std::fopen(("results/bank_path_variants" + std::string(cop == "gaussian" ? "" : "_" + cop) + ".csv").c_str(), "w");
    std::fprintf(out, "variant,scenario,t,truth,bank");
    for (std::size_t k = 0; k < gains.size(); ++k) std::fprintf(out, ",arm_%zu", k);
    for (std::size_t k = 0; k < gains.size(); ++k) std::fprintf(out, ",w_%zu", k);
    std::fprintf(out, "\n");
    for (const auto& [name, delta] : std::vector<std::pair<std::string, double>>{{"softmax", 0.0}, {"fixed_share", 3e-4}})
        for (const std::string sc : {"drift", "jump"}) {
            std::unique_ptr<Dependence> dep;
            if (sc == "drift") dep = std::make_unique<DriftDependence>(-0.5, 0.8, 5000);
            else dep = std::make_unique<JumpDependence>(0.2, 0.8, 2500);
            auto c = dgp::make_copula(cop);
            CopulaSource src(*dep, *c, seed);
            std::vector<std::unique_ptr<DependenceEstimator>> m;
            for (double g : gains) {
                RecursiveMPL::Config cfg{};
                cfg.margin = RecursiveMPL::Margin::ExactNormal; cfg.gain = g; cfg.startup = true;
                m.push_back(std::make_unique<RecursiveMPL>(cfg, "m"));
            }
            ForgettingBank bank(std::move(m), ForgettingBank::Config{.loss = ForgettingBank::Loss::Log, .eta = 1.0, .delta = delta}, "b");
            for (std::size_t t = 0; t < 5000; ++t) {
                const auto smp = src.next();
                std::fprintf(out, "%s,%s,%zu,%.5f,%.5f", name.c_str(), sc.c_str(), t, smp.rho_s, bank.estimate());
                for (std::size_t k = 0; k < gains.size(); ++k) std::fprintf(out, ",%.5f", bank.model(k).estimate());
                for (std::size_t k = 0; k < gains.size(); ++k) std::fprintf(out, ",%.5g", bank.weight(k));
                std::fprintf(out, "\n");
                bank.update(smp.x, smp.y);
            }
        }
    std::fclose(out);
}
