// Compare fixed, MFF and VFF strategies on matched streams, scoring forecasts before each update.
// Usage: COPULA=<family> strategy_experiments [seeds=60]; LIVE=1 uses the TCP stream.
// See the section run.sh scripts for study flags and output files.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../../src/special_functions.hpp"
#include "../../src/estimators/device.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "../../src/estimators/registry.hpp"
#include "simulation.hpp"
#include "simulation_reports.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace {
using thesis::family_of; using thesis::H2Table; using thesis::kK;
using strategy_experiments::Run;
using strategy_experiments::Simulation;
using strategy_experiments::StrategyOptions;
using strategy_experiments::ForecastComparisonReport;
using strategy_experiments::MffWeightPathReport;

// Strategies: five fixed arms, MFF, then one VFF per rate. Default rates 0.02 / 0.1 named vff_slow / vff_fast (the
// original outputs); VFF_RATES=a,b,... names them vff_<a>, ... (sensitivity runs; write under TAG=...).
std::vector<std::string> kStrat = thesis::strategy_names();
std::vector<double> kRates = thesis::vff_rates();

// The standard strategy comparison: one Simulation per scenario and seed, collated by ForecastComparisonReport.
// MIXDENS=1 scores the weighted density mixture. Its KLL log loss also checks the regret calculation.
// WEIGHTS=1 replays the same runs and writes the MFF weight paths, with plain Hedge (delta 0) alongside.
class Experiment {
public:
    explicit Experiment(const Run& run) : run_(run) {}
    int run();
    int run_weight_paths();

private:
    const Run& run_;
};

int Experiment::run() {
    StrategyOptions options{std::getenv("MIXDENS") != nullptr, false, std::getenv("DUMP_SEED"),
                            std::getenv("DUMP_SCEN") ? std::getenv("DUMP_SCEN") : "jump"};
    if (options.mixdens && !run_.tag) { std::fprintf(stderr, "MIXDENS needs TAG\n"); return 1; }
    options.paths = std::getenv("PATHS") != nullptr || options.dump_seed;   // per-tick paths / dump only; the other outputs are left alone
    ForecastComparisonReport results(run_, options, kStrat);
    if (!results.ok()) return strategy_experiments::missing_results_folder();

    for (const std::string sc : {"constant", "drift", "jump"}) {
        if (options.dump_seed && sc != options.dump_scenario) continue;
        results.begin_scenario(sc);
        for (int sd = 0; sd < run_.seeds; ++sd) {
            const unsigned seed = first_seed() + sd;
            if (options.dump_seed && int(seed) != std::atoi(options.dump_seed)) continue;
            results.add(sd, Simulation(run_.cop, run_.h2, sc, seed, kRates, {.mixture_density = options.mixdens}).run());
        }
        results.end_scenario();
    }
    return 0;
}

int Experiment::run_weight_paths() {
    MffWeightPathReport results(run_);
    if (!results.ok()) return strategy_experiments::missing_results_folder();
    for (const std::string sc : {"constant", "drift", "jump"}) {
        results.begin_scenario(sc);
        for (int sd = 0; sd < run_.seeds; ++sd) {
            const Simulation simulation(run_.cop, run_.h2, sc, first_seed() + sd, {}, {.scores = false, .weight_paths = true});
            results.add(simulation.run());
        }
        results.end_scenario();
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : thesis::kSeedCount;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    const char* tag = std::getenv("TAG");
    const std::string sfx = (tag ? std::string("_") + tag : std::string()) + (cop == "gaussian" ? "" : "_" + cop);
    if (const char* r = std::getenv("VFF_RATES")) {
        kRates.clear(); kStrat.resize(kK + 1);
        for (const auto& rate : csv_list(r)) {
            kRates.push_back(std::atof(rate.c_str()));
            kStrat.push_back("vff_" + rate);
        }
    }
    H2Table h2{.F = family_of(cop)};
    const Run run{seeds, cop, sfx, tag, h2};
    if (std::getenv("VALIDATE")) return strategy_experiments::run_validate(run);
    if (std::getenv("FAMSWITCH")) return strategy_experiments::run_family_switching_study(run);
    if (std::getenv("SCORESFAM")) return strategy_experiments::run_family_score_comparison(run);
    if (std::getenv("REGRETFAM")) return strategy_experiments::run_regretfam(run);
    if (std::getenv("REGRET")) return strategy_experiments::run_regret(run);
    if (std::getenv("WEIGHTS")) return Experiment(run).run_weight_paths();
    return Experiment(run).run();
}
