// Compare fixed, MFF and VFF strategies on matched streams, scoring forecasts before each update.
// Usage: strategy_experiments <mode> [seeds=60] (modes in kUsage below); LIVE=1 uses the TCP stream.
// See the section run.sh scripts for the output files.

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
    int run(bool mixture);
    int run_weight_paths();

private:
    const Run& run_;
};

int Experiment::run(bool mixture) {
    StrategyOptions options{mixture, false, std::getenv("DUMP_SEED"),
                            std::getenv("DUMP_SCEN") ? std::getenv("DUMP_SCEN") : "jump"};
    if (options.mixdens && !run_.tag) { std::fprintf(stderr, "mixture needs TAG\n"); return 1; }
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

constexpr const char* kUsage =
    "usage: strategy_experiments <mode> [seeds=60]\n"
    "  strategies        fixed, MFF and VFF forecasts (strategies_seeds); PATHS=1 adds per-tick paths\n"
    "  mixture           the same, MFF density scored as the weighted mixture (needs TAG)\n"
    "  weights           MFF weight paths (strategies_weights)\n"
    "  regret            MFF regret against the best fixed and switching sequences\n"
    "  regret-family     regret across working families\n"
    "  score-family      squared vs log loss across working families (needs TAG)\n"
    "  family-switching  family-augmented MFF (FS_GENS, FS_H2, FS_EST, FS_TAG)\n"
    "  validate          table and density self-checks\n"
    "options (environment): COPULA, TAG, VFF_RATES, SEED_BASE, DUMP_SEED/DUMP_SCEN, LIVE=1\n";

int usage() { std::fputs(kUsage, stderr); return 2; }

int main(int argc, char** argv) {
    for (const char* old : {"VALIDATE", "FAMSWITCH", "SCORESFAM", "REGRETFAM", "REGRET", "WEIGHTS", "MIXDENS"})
        if (std::getenv(old)) { std::fprintf(stderr, "%s=1 is now a mode argument\n", old); return usage(); }
    if (argc < 2 || argc > 3) return usage();
    const std::string mode = argv[1];
    const int seeds = argc > 2 ? std::atoi(argv[2]) : thesis::kSeedCount;
    if (seeds < 1) return usage();
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
    if (mode == "strategies") return Experiment(run).run(false);
    if (mode == "mixture") return Experiment(run).run(true);
    if (mode == "weights") return Experiment(run).run_weight_paths();
    if (mode == "regret") return strategy_experiments::run_regret(run);
    if (mode == "regret-family") return strategy_experiments::run_regretfam(run);
    if (mode == "score-family") return strategy_experiments::run_family_score_comparison(run);
    if (mode == "family-switching") return strategy_experiments::run_family_switching_study(run);
    if (mode == "validate") return strategy_experiments::run_validate(run);
    std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
    return usage();
}
