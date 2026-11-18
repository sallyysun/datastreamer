// Working-family scoring rules and family identification at matched Spearman dependence.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "simulation_reports.hpp"
#include "strategy_runs.hpp"
#include "working_families.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

using thesis::H2Table;
using thesis::Win;
using thesis::checkpoint;
using thesis::kBurn;
using thesis::kChange;
using thesis::kK;
using thesis::kN;
using thesis::windows;

namespace {

constexpr std::size_t kFamilies = WorkingFamilies::kCount;

double squared(double x) { return x * x; }

// One forecaster's scores at every tick: squared Spearman error, excess log loss, and squared Hellinger at checkpoints.
struct ScoreTrace {
    std::vector<double> squared_error = std::vector<double>(kN), excess = std::vector<double>(kN), h2 = std::vector<double>(kN, NAN);

    void record_error_and_excess_loss(std::size_t t, double forecast, double truth, double truth_log_density, double mixture_log_density) {
        squared_error[t] = squared(forecast - truth);
        excess[t] = truth_log_density - mixture_log_density;
    }
};

// ---- Score comparison (SCORESFAM) ----

enum class LossRule { Squared, Log };

const char* rule_name(LossRule rule) { return rule == LossRule::Log ? "log" : "squared"; }

// Each fixed-memory arm's forecast before every update, and readiness, recorded once per seed and estimator.
struct FixedMemoryForecasts {
    std::vector<std::vector<double>> rho;   // [k][t]
    std::vector<char> ready;                // [t]
};

FixedMemoryForecasts record_fixed_memory_forecasts(const char* estimator, const StudySeries& s) {
    FixedMemoryForecasts forecasts{std::vector<std::vector<double>>(kK, std::vector<double>(kN)), std::vector<char>(kN)};
    FixedMemoryArms arms(estimator);
    for (std::size_t t = 0; t < kN; ++t) {
        for (std::size_t k = 0; k < kK; ++k) forecasts.rho[k][t] = arms.estimate(k);
        forecasts.ready[t] = arms.ready();
        arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
    }
    return forecasts;
}

// One MFF learning rate and loss rule, scored at every tick.
struct ScoreVariant {
    LossRule rule;
    thesis::Mff mixture;
    ScoreTrace scores;
};

struct ScoreComparison {
    std::vector<ScoreVariant> variants;
    long clamped_ticks = 0, score_nonfinite = 0, eval_nonfinite = 0;
};

// Every learning rate and both loss rules for one estimator and working family, from the same arm forecasts.
// Forecasts are scored at the true PIT; the weights update at the estimated PIT.
ScoreComparison compare_mff_loss_rules(const StudySeries& s, const std::string& scenario, const FixedMemoryForecasts& arms, std::size_t working,
                                    const WorkingFamilies& families, FamilyHellinger& hellinger, const std::vector<double>& generating_parameter,
                                    const std::vector<double>& truth_log_density, const std::vector<double>& etas, bool with_h2) {
    const auto family = WorkingFamilies::family[working];
    const auto& inverse = families.inverse[working];
    ScoreComparison result;
    for (double eta : etas)
        for (LossRule rule : {LossRule::Squared, LossRule::Log})
            result.variants.push_back({rule, thesis::Mff(copula::Family::Gaussian, kK, eta), {}});

    std::vector<double> squared_losses(kK), log_losses(kK), log_density(kK), rho(kK);
    for (std::size_t t = 0; t < kN; ++t) {
        const bool update = t >= thesis::kMffWarmup && arms.ready[t];
        const double moment = 12.0 * (s.estimated_pit_x[t] - 0.5) * (s.estimated_pit_y[t] - 0.5);
        for (std::size_t k = 0; k < kK; ++k) {
            rho[k] = arms.rho[k][t];
            log_density[k] = copula::log_density(family, s.true_pit_x[t], s.true_pit_y[t], families.evaluation_parameter(working, rho[k]));
            if (!std::isfinite(log_density[k])) { ++result.eval_nonfinite; log_density[k] = -INFINITY; }
            if (!update && t < kBurn) continue;
            const double parameter = inverse(rho[k]);
            const bool clamped = working == 0 ? std::fabs(parameter) >= copula::Gaussian::kMaxParam : inverse.clamped(rho[k]);
            if (clamped && t >= kBurn) ++result.clamped_ticks;
            if (update) {
                squared_losses[k] = squared(moment - rho[k]);
                log_losses[k] = -copula::log_density(family, s.estimated_pit_x[t], s.estimated_pit_y[t], parameter);
            }
        }
        const bool score_h2 = with_h2 && checkpoint(scenario, t);
        for (auto& v : result.variants) {
            v.scores.record_error_and_excess_loss(t, v.mixture.forecast(rho.data()), s.true_spearman[t], truth_log_density[t],
                            v.mixture.mixture_log_density(log_density.data()));
            if (score_h2) v.scores.h2[t] = hellinger.single_working_family(working, generating_parameter[t], rho.data(), v.mixture.w.data(), kK);
        }
        if (update) {
            result.score_nonfinite += thesis::guard_mff_losses(log_losses);
            for (auto& v : result.variants) v.mixture.update(v.rule == LossRule::Log ? log_losses : squared_losses);
        }
    }
    return result;
}

// ---- Family switching (FAMSWITCH) ----

struct Generator {
    const char* name;
    std::size_t before, after;   // working-family index before and after tick kChange
    bool switches() const { return before != after; }
};

constexpr std::array<Generator, 7> kGenerators{{{"gaussian", 0, 0}, {"t", 1, 1}, {"clayton", 2, 2}, {"gumbel", 3, 3},
                                                 {"gauss2t", 0, 1}, {"gauss2clayton", 0, 2}, {"clayton2gumbel", 2, 3}}};

struct FamilySwitchSeed {
    ScoreTrace family_aware, gaussian_only;
    std::vector<std::array<double, kFamilies>> family_weight = std::vector<std::array<double, kFamilies>>(kN);   // [t][family]
    long guarded_losses = 0;
};

// One seed: family-aware MFF (every working family x memory, arm i = family i / kK, memory i % kK) against
// Gaussian-only MFF (the first kK arms), both fed the same arm forecasts.
FamilySwitchSeed compare_family_mixtures_for_seed(const Generator& generator, const StudySeries& s, const std::string& estimator,
                                        const WorkingFamilies& families, FamilyHellinger& hellinger, bool with_h2) {
    constexpr std::size_t kArms = kFamilies * kK;
    FamilySwitchSeed result;
    FixedMemoryArms arms(estimator);
    thesis::Mff family_aware(copula::Family::Gaussian, kArms), gaussian_only(copula::Family::Gaussian, kK);
    double rho[kK], update_parameter[kFamilies][kK];
    std::array<double, kArms> arm_rho{}, log_density{};
    std::array<int, kArms> arm_family{};
    for (std::size_t i = 0; i < kArms; ++i) arm_family[i] = int(i / kK);
    std::vector<double> family_losses(kArms), gaussian_losses(kK);
    const char* h2_scenario = generator.switches() ? "jump" : "constant";
    for (std::size_t t = 0; t < kN; ++t) {
        arms.forecasts(rho);
        const bool ready = t >= thesis::kMffWarmup && arms.ready();
        const auto truth_family = WorkingFamilies::family[t < kChange ? generator.before : generator.after];
        const double truth_parameter = copula::param_from_spearman(truth_family, thesis::kConstantRhoS);
        const double truth_log_density = copula::log_density(truth_family, s.true_pit_x[t], s.true_pit_y[t], truth_parameter);
        for (std::size_t w = 0; w < kFamilies; ++w)
            for (std::size_t k = 0; k < kK; ++k) {
                update_parameter[w][k] = families.inverse[w](rho[k]);
                log_density[w * kK + k] = copula::log_density(WorkingFamilies::family[w], s.true_pit_x[t], s.true_pit_y[t],
                                                              families.evaluation_parameter(w, rho[k]));
            }
        for (std::size_t i = 0; i < kArms; ++i) arm_rho[i] = rho[i % kK];

        const double truth = s.true_spearman[t];
        result.family_aware.record_error_and_excess_loss(t, family_aware.forecast(arm_rho.data()), truth, truth_log_density,
                                   family_aware.mixture_log_density(log_density.data()));
        result.gaussian_only.record_error_and_excess_loss(t, gaussian_only.forecast(rho), truth, truth_log_density,
                                    gaussian_only.mixture_log_density(log_density.data()));   // the Gaussian arms come first
        if (with_h2 && checkpoint(h2_scenario, t)) {
            result.family_aware.h2[t] = hellinger.mixed_families(truth_family, truth_parameter, arm_family.data(), arm_rho.data(),
                                                                 family_aware.w.data(), family_aware.size());
            result.gaussian_only.h2[t] = hellinger.mixed_families(truth_family, truth_parameter, arm_family.data(), arm_rho.data(),
                                                                  gaussian_only.w.data(), gaussian_only.size());
        }
        for (std::size_t w = 0; w < kFamilies; ++w) {
            double total = 0;
            for (std::size_t k = 0; k < kK; ++k) total += family_aware.w[w * kK + k];
            result.family_weight[t][w] = total;
        }
        if (ready) {
            for (std::size_t w = 0; w < kFamilies; ++w)
                for (std::size_t k = 0; k < kK; ++k)
                    family_losses[w * kK + k] = -copula::log_density(WorkingFamilies::family[w], s.estimated_pit_x[t],
                                                                     s.estimated_pit_y[t], update_parameter[w][k]);
            std::copy_n(family_losses.begin(), kK, gaussian_losses.begin());
            result.guarded_losses += thesis::guard_mff_losses(family_losses);
            result.guarded_losses += thesis::guard_mff_losses(gaussian_losses);
            family_aware.update(family_losses);
            gaussian_only.update(gaussian_losses);
        }
        arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
    }
    return result;
}

// How quickly the target family's weight reaches 50% and 90%, counted from tick 0 (constant) or kChange (switch).
struct FamilyIdentification {
    long t50 = -1, t90 = -1, first_incorrect = -1;
    double share_ge50_after, pre_weight_target;
    bool incorrect_majority = false;
};

FamilyIdentification summarize_family_identification(const Generator& generator, const FamilySwitchSeed& seed) {
    const std::size_t target = generator.after, start = generator.switches() ? kChange : 0;
    const auto& weight = seed.family_weight;
    FamilyIdentification id;
    long after = 0, majority = 0;
    for (std::size_t t = start; t < kN; ++t) {
        const long since = long(t - start);
        const double w = weight[t][target];
        if (id.t50 < 0 && w >= 0.5) id.t50 = since;
        if (id.t90 < 0 && w >= 0.9) id.t90 = since;
        if (id.t50 >= 0 && since >= id.t50) {
            ++after;
            if (w >= 0.5) ++majority;
            for (std::size_t f = 0; f < kFamilies; ++f)
                if (f != target && weight[t][f] >= 0.5 && !id.incorrect_majority) { id.incorrect_majority = true; id.first_incorrect = since; }
        }
    }
    id.share_ge50_after = after ? double(majority) / after : NAN;
    id.pre_weight_target = generator.switches() ? weight[kChange - 1][target] : weight[0][target];
    return id;
}

std::vector<Win> family_evaluation_windows(const Generator& generator) {
    std::vector<Win> wins{{"whole", kBurn, kN}};
    if (generator.switches())
        wins.insert(wins.end(), {{"pre", kBurn, kChange}, {"rec_0_100", kChange, kChange + 100}, {"late", kChange + 100, kN}, {"post", kChange, kN}});
    return wins;
}

void write_switch_seed(std::FILE* seeds_file, std::FILE* ident_file, const Generator& generator, unsigned seed,
                       const std::vector<Win>& wins, const FamilySwitchSeed& result, bool with_h2) {
    for (const auto& w : wins)
        for (const auto& [method, trace] : {std::pair{"family_mff", &result.family_aware}, std::pair{"gaussian_mff", &result.gaussian_only}}) {
            const auto [rmse, excess, h2_score] = summarize_window(w, trace->squared_error, trace->excess, trace->h2);
            if (with_h2) std::fprintf(seeds_file, "%s,%u,%s,%s,%.8f,%.6f,%.6f\n", generator.name, seed, method, w.name, rmse, excess, h2_score);
            else std::fprintf(seeds_file, "%s,%u,%s,%s,%.8f,%.6f\n", generator.name, seed, method, w.name, rmse, excess);
        }
    const FamilyIdentification id = summarize_family_identification(generator, result);
    std::fprintf(ident_file, "%s,%u,%s,%s,%ld,%ld,%.6f,%d,%ld,%.6f,%ld\n", generator.name, seed, generator.switches() ? "switch" : "initial",
                 WorkingFamilies::name[generator.after], id.t50, id.t90, id.share_ge50_after, int(id.incorrect_majority),
                 id.first_incorrect, id.pre_weight_target, result.guarded_losses);
}

}  // namespace

// Compare squared and log loss across working copulas, using the same KLL inputs and arms.
int run_family_score_comparison(const Run& run) {
    const unsigned base = first_seed();
    const bool with_h2 = std::getenv("NOH2") == nullptr;
    std::vector<double> etas{1.0};
    if (const char* es = std::getenv("ETAS")) {
        etas.clear();
        for (const auto& eta : csv_list(es)) etas.push_back(std::atof(eta.c_str()));
    }
    if (!run.tag) { std::fprintf(stderr, "SCORESFAM needs TAG\n"); return 1; }
    const WorkingFamilies families;
    FamilyHellinger hellinger(run.h2, families.inverse, with_h2);
    std::FILE* seeds_file = open_result("scoresfam_seeds" + run.sfx);
    std::FILE* diag_file = open_result("scoresfam_diag" + run.sfx);
    if (!seeds_file || !diag_file) return missing_results_folder();
    std::fprintf(seeds_file, "gen,scenario,estimator,working,score,eta,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(diag_file, "gen,scenario,estimator,working,seed,arm_clamp_ticks,score_nonfinite,eval_nonfinite\n");
    const char* cop = run.cop.c_str();
    constexpr const char* kEstimators[3] = {"rmpl", "xiao", "hermite"};
    for (const std::string scenario : {"constant", "drift", "jump"}) {
        const auto wins = windows(scenario);
        for (int sd = 0; sd < run.seeds; ++sd) {
            const unsigned seed = base + sd;
            const StudySeries s = load_study_series(run.cop, scenario, seed, kN);
            const auto generating_parameter = generating_parameters(run.cop, s);
            std::vector<double> truth_log_density(kN);
            for (std::size_t t = 0; t < kN; ++t)
                truth_log_density[t] = copula::log_density(run.h2.F, s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t]);
            std::vector<FixedMemoryForecasts> forecasts;
            for (const char* estimator : kEstimators) forecasts.push_back(record_fixed_memory_forecasts(estimator, s));
            for (std::size_t working = 0; working < kFamilies; ++working)
                for (std::size_t e = 0; e < 3; ++e) {
                    const auto comparison = compare_mff_loss_rules(s, scenario, forecasts[e], working, families, hellinger, generating_parameter,
                                                                truth_log_density, etas, with_h2);
                    const char* family = WorkingFamilies::name[working];
                    for (const auto& v : comparison.variants)
                        for (const auto& w : wins) {
                            const auto [rmse, excess, h2_score] = summarize_window(w, v.scores.squared_error, v.scores.excess, v.scores.h2);
                            std::fprintf(seeds_file, "%s,%s,%s,%s,%s,%g,%u,%s,%.8f,%.6f,%.6f\n", cop, scenario.c_str(), kEstimators[e], family,
                                         rule_name(v.rule), v.mixture.eta, seed, w.name, rmse, excess, h2_score);
                        }
                    std::fprintf(diag_file, "%s,%s,%s,%s,%u,%ld,%ld,%ld\n", cop, scenario.c_str(), kEstimators[e], family, seed,
                                 comparison.clamped_ticks, comparison.score_nonfinite, comparison.eval_nonfinite);
                }
        }
        std::fflush(stdout);
    }
    std::fclose(seeds_file);
    std::fclose(diag_file);
    return 0;
}

// Compare family-aware and Gaussian-only MFF at matched dependence. Families switch at tick 2500.
// FS_GENS picks generators, FS_EST the estimator (rmpl by default), FS_H2 adds squared Hellinger, FS_TAG names outputs.
int run_family_switching_study(const Run& run) {
    const WorkingFamilies families;
    const std::string fx = std::getenv("FS_TAG") ? std::string("_") + std::getenv("FS_TAG") : std::string();
    std::FILE* seeds_file = open_result("famsw_seeds" + fx);
    std::FILE* ident_file = open_result("famsw_ident" + fx);
    std::FILE* path_file = open_result("famsw_path" + fx);
    if (!seeds_file || !ident_file || !path_file) return missing_results_folder();
    const bool with_h2 = std::getenv("FS_H2") != nullptr;
    std::fprintf(seeds_file, with_h2 ? "gen,seed,method,window,rmse,excess_x1000,h2_x1000\n" : "gen,seed,method,window,rmse,excess_x1000\n");
    FamilyHellinger hellinger(run.h2, families.inverse, with_h2);
    std::fprintf(ident_file, "gen,seed,kind,target,t50,t90,share_ge50_after,incorrect_majority,first_incorrect,pre_weight_target,n_guard\n");
    std::fprintf(path_file, "gen,t,family,mean,sd\n");
    const unsigned base = first_seed();
    const std::string estimator = std::getenv("FS_EST") ? std::getenv("FS_EST") : "rmpl";
    std::vector<Generator> generators;
    if (const char* names = std::getenv("FS_GENS")) {
        for (const auto& name : csv_list(names))
            for (const auto& g : kGenerators) if (name == g.name) generators.push_back(g);
    } else generators.assign(kGenerators.begin(), kGenerators.end());

    for (const auto& generator : generators) {
        const auto wins = family_evaluation_windows(generator);
        std::vector<double> weight_sum(kFamilies * kN, 0.0), weight_sum_sq(kFamilies * kN, 0.0);
        for (int sd = 0; sd < run.seeds; ++sd) {
            const unsigned seed = base + sd;
            const StudySeries s = load_study_series(generator.name, "constant", seed, kN);
            const FamilySwitchSeed result = compare_family_mixtures_for_seed(generator, s, estimator, families, hellinger, with_h2);
            write_switch_seed(seeds_file, ident_file, generator, seed, wins, result, with_h2);
            for (std::size_t t = 0; t < kN; ++t)
                for (std::size_t f = 0; f < kFamilies; ++f) {
                    const double w = result.family_weight[t][f];
                    weight_sum[f * kN + t] += w;
                    weight_sum_sq[f * kN + t] += w * w;
                }
        }
        for (std::size_t t = 0; t < kN; t += 10)
            for (std::size_t f = 0; f < kFamilies; ++f) {
                const auto [mean, sd] = seed_mean_sd(weight_sum[f * kN + t], weight_sum_sq[f * kN + t], run.seeds);
                std::fprintf(path_file, "%s,%zu,%s,%.6f,%.6f\n", generator.name, t, WorkingFamilies::name[f], mean, sd);
            }
    }
    std::fclose(seeds_file);
    std::fclose(ident_file);
    std::fclose(path_file);
    return 0;
}

}  // namespace strategy_experiments
