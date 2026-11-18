// Regret against the best fixed arm and the best one- or two-switch sequences.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"
#include "working_families.hpp"

inline constexpr int kSwitchBudgets = 3;

struct RegretOracle {
    std::vector<std::array<double, kSwitchBudgets>> prefix_loss;
    double fixed_loss, one_switch_loss;
    std::size_t fixed_arm, switch_index, before_arm, after_arm;
};

inline RegretOracle regret_oracle(const std::vector<std::vector<double>>& losses) {
    if (losses.empty() || losses.front().empty()) {
        throw std::invalid_argument("regret oracle needs losses for at least one tick and arm");
    }
    const std::size_t ticks = losses.size(), arms = losses.front().size();
    constexpr double infinity = std::numeric_limits<double>::infinity();
    RegretOracle oracle{};
    oracle.prefix_loss.resize(ticks);
    oracle.one_switch_loss = infinity;

    // State is the minimum loss for each final arm and exact switch count.
    std::vector<double> previous(kSwitchBudgets * arms, infinity), current(previous.size());
    for (std::size_t t = 0; t < ticks; ++t) {
        double best = infinity;
        for (int switches = 0; switches < kSwitchBudgets; ++switches) {
            double change = infinity;
            if (switches > 0) {
                for (std::size_t arm = 0; arm < arms; ++arm) {
                    change = std::min(change, previous[(switches - 1) * arms + arm]);
                }
            }
            for (std::size_t arm = 0; arm < arms; ++arm) {
                const auto index = switches * arms + arm;
                const double start = t == 0 && switches == 0 ? 0.0 : previous[index];
                current[index] = losses[t][arm] + std::min(start, change);
                best = std::min(best, current[index]);
            }
            oracle.prefix_loss[t][switches] = best;
        }
        previous.swap(current);
    }

    // Prefix sums identify the static arm and the first best one-switch path.
    std::vector<std::vector<double>> cumulative(arms, std::vector<double>(ticks + 1, 0.0));
    for (std::size_t arm = 0; arm < arms; ++arm) {
        for (std::size_t t = 0; t < ticks; ++t) {
            cumulative[arm][t + 1] = cumulative[arm][t] + losses[t][arm];
        }
        if (cumulative[arm][ticks] < cumulative[oracle.fixed_arm][ticks]) oracle.fixed_arm = arm;
    }
    oracle.fixed_loss = cumulative[oracle.fixed_arm][ticks];

    for (std::size_t t = 0; t <= ticks; ++t) {
        for (std::size_t before = 0; before < arms; ++before) {
            for (std::size_t after = 0; after < arms; ++after) {
                const double loss = cumulative[before][t] + cumulative[after][ticks] - cumulative[after][t];
                if (loss < oracle.one_switch_loss - 1e-12) {
                    oracle.one_switch_loss = loss;
                    oracle.switch_index = t;
                    oracle.before_arm = before;
                    oracle.after_arm = after;
                }
            }
        }
    }
    return oracle;
}

namespace strategy_experiments {

using thesis::kK;
using thesis::kMem;
using thesis::kN;

namespace {

constexpr std::array<double, 2> kShareRates{0.0, 1e-4};   // plain Hedge, then the study's fixed share

struct MixtureLoss {
    double mix = 0, point = 0;
};

// One seed's scored ticks: each arm's guarded log loss, and each share rate's mixture (and point-forecast) loss,
// all taken before that tick's weight update.
struct RegretSeed {
    std::vector<std::vector<double>> arm_losses;                  // [scored tick][arm]
    std::vector<std::array<MixtureLoss, kShareRates.size()>> mixture;   // [scored tick][share rate]
    std::vector<std::size_t> ticks;                               // series tick of each scored row
    long nonfinite = 0, clamped = 0;
};

// arm_loss(rho, u, v, clamped) is one arm's log loss at the estimated PIT and marks a clamped parameter.
template <class ArmLoss>
RegretSeed collect_regret_losses(const StudySeries& s, const std::string& estimator, copula::Family family, ArmLoss arm_loss,
                                 bool point_losses) {
    RegretSeed seed;
    FixedMemoryArms arms(estimator);
    std::array<thesis::Mff, kShareRates.size()> mixtures{thesis::Mff(family, kK, 1.0, kShareRates[0]),
                                                         thesis::Mff(family, kK, 1.0, kShareRates[1])};
    std::vector<double> rho(kK), losses(kK);
    for (std::size_t t = 0; t < kN; ++t) {
        if (t >= thesis::kMffWarmup && arms.ready()) {
            const double u = s.estimated_pit_x[t], v = s.estimated_pit_y[t];
            arms.forecasts(rho.data());
            bool clamped = false;
            for (std::size_t k = 0; k < kK; ++k) losses[k] = arm_loss(rho[k], u, v, clamped);
            seed.clamped += clamped;
            seed.nonfinite += thesis::guard_mff_losses(losses);
            std::array<MixtureLoss, kShareRates.size()> row{};
            for (std::size_t i = 0; i < mixtures.size(); ++i) {
                row[i].mix = mixtures[i].mixture_loss(losses.data());
                if (point_losses) {
                    const double forecast = mixtures[i].forecast(rho.data());
                    row[i].point = -copula::log_density(copula::Family::Gaussian, u, v,
                                                        copula::param_from_spearman(copula::Family::Gaussian, forecast));
                }
                mixtures[i].update(losses);
            }
            seed.arm_losses.push_back(losses);
            seed.mixture.push_back(row);
            seed.ticks.push_back(t);
        }
        arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
    }
    return seed;
}

// Series tick of the one-switch comparator's switch, or 0 when its best path does not switch.
std::size_t switch_tick(const RegretOracle& oracle, const RegretSeed& seed) {
    if (oracle.before_arm == oracle.after_arm) return 0;
    return seed.ticks[std::min(oracle.switch_index, seed.ticks.size() - 1)];
}

double cumulative_mixture_loss(const RegretSeed& seed, std::size_t rate) {
    double total = 0;
    for (const auto& row : seed.mixture) total += row[rate].mix;
    return total;
}

// Sum and sum of squares across seeds at every tick.
struct TickMoments {
    std::vector<double> sum = std::vector<double>(kN, 0.0), sum_sq = std::vector<double>(kN, 0.0);
    void add(std::size_t t, double x) { sum[t] += x; sum_sq[t] += x * x; }
};

// Cumulative regret paths by switch budget for one estimator and share rate.
struct RegretPaths {
    std::array<TickMoments, kSwitchBudgets> mixture, point;
};

}  // namespace

// Compare family-specific mixture regret against the best arm and the best one- or two-switch paths.
int run_regretfam(const Run& run) {
    std::FILE* seeds_file = open_result("regretfam_seeds" + run.sfx);
    if (!seeds_file) return missing_results_folder();
    std::fprintf(seeds_file, "gen,scenario,working,delta,seed,T,t0,loss_mix,best0,arm0,best1,best2,sw_t,sw_from,sw_to,n_nonfinite,n_clamp\n");
    const WorkingFamilies families;
    for (const std::string scenario : {"constant", "drift", "jump"}) {
        for (int sd = 0; sd < run.seeds; ++sd) {
            const unsigned seed = first_seed() + sd;
            const StudySeries s = load_study_series(run.cop, scenario, seed, kN);
            for (std::size_t working = 0; working < WorkingFamilies::kCount; ++working) {
                const auto& inverse = families.inverse[working];
                const auto family = WorkingFamilies::family[working];
                auto arm_loss = [&](double rho, double u, double v, bool& clamped) {
                    clamped = clamped || inverse.clamped(rho);
                    return -copula::log_density(family, u, v, inverse(rho));
                };
                const RegretSeed result = collect_regret_losses(s, "rmpl", family, arm_loss, false);
                const auto oracle = regret_oracle(result.arm_losses);
                for (std::size_t rate = 0; rate < kShareRates.size(); ++rate)
                    std::fprintf(seeds_file, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g,%ld,%ld\n", run.cop.c_str(),
                                 scenario.c_str(), WorkingFamilies::name[working], kShareRates[rate], seed, result.ticks.size(),
                                 result.ticks.front(), cumulative_mixture_loss(result, rate), oracle.fixed_loss, kMem[oracle.fixed_arm],
                                 oracle.one_switch_loss, oracle.prefix_loss.back()[2], switch_tick(oracle, result),
                                 kMem[oracle.before_arm], kMem[oracle.after_arm], result.nonfinite, result.clamped);
            }
        }
    }
    std::fclose(seeds_file);
    return 0;
}

// Compare Gaussian mixture and point-forecast regret against static and switching oracles.
int run_regret(const Run& run) {
    std::FILE* seeds_file = open_result("regret_seeds" + run.sfx);
    std::FILE* paths_file = open_result("regret_paths" + run.sfx);
    if (!seeds_file || !paths_file) return missing_results_folder();
    std::fprintf(seeds_file, "copula,scenario,estimator,delta,seed,T,t0,loss_mix,loss_point,best0,arm0,best1,best2,sw_t,sw_from,sw_to\n");
    std::fprintf(paths_file, "copula,scenario,estimator,delta,loss,m,t,mean,sd\n");
    const auto& estimators = thesis::estimators();
    auto arm_loss = [](double rho, double u, double v, bool&) {
        return -copula::log_density(copula::Family::Gaussian, u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
    };
    for (const std::string scenario : {"constant", "drift", "jump"}) {
        std::vector<std::array<RegretPaths, kShareRates.size()>> paths(estimators.size());
        std::vector<char> scored(kN, 0);
        for (int sd = 0; sd < run.seeds; ++sd) {
            const unsigned seed = first_seed() + sd;
            const StudySeries s = load_study_series(run.cop, scenario, seed, kN);
            for (std::size_t e = 0; e < estimators.size(); ++e) {
                const RegretSeed result = collect_regret_losses(s, estimators[e], copula::Family::Gaussian, arm_loss, true);
                for (std::size_t t : result.ticks) scored[t] = 1;
                const auto oracle = regret_oracle(result.arm_losses);
                for (std::size_t rate = 0; rate < kShareRates.size(); ++rate) {
                    auto& rate_paths = paths[e][rate];
                    double mix = 0, point = 0;
                    for (std::size_t i = 0; i < result.ticks.size(); ++i) {
                        mix += result.mixture[i][rate].mix;
                        point += result.mixture[i][rate].point;
                        for (int m = 0; m < kSwitchBudgets; ++m) {
                            rate_paths.mixture[m].add(result.ticks[i], mix - oracle.prefix_loss[i][m]);
                            rate_paths.point[m].add(result.ticks[i], point - oracle.prefix_loss[i][m]);
                        }
                    }
                    std::fprintf(seeds_file, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g\n", run.cop.c_str(), scenario.c_str(),
                                 estimators[e].c_str(), kShareRates[rate], seed, result.ticks.size(), result.ticks.front(), mix, point,
                                 oracle.fixed_loss, kMem[oracle.fixed_arm], oracle.one_switch_loss, oracle.prefix_loss.back()[2],
                                 switch_tick(oracle, result), kMem[oracle.before_arm], kMem[oracle.after_arm]);
                }
            }
        }
        for (std::size_t e = 0; e < estimators.size(); ++e)
            for (std::size_t rate = 0; rate < kShareRates.size(); ++rate) {
                const auto& rate_paths = paths[e][rate];
                for (const auto& [loss, by_budget] : {std::pair{"mix", &rate_paths.mixture}, std::pair{"point", &rate_paths.point}})
                    for (int m = 0; m < kSwitchBudgets; ++m) {
                        const TickMoments& path = (*by_budget)[m];
                        for (std::size_t t = 0; t < kN; ++t) {
                            if (!scored[t] || (t % 10 != 9 && t != kN - 1)) continue;
                            const auto [mean, sd] = seed_mean_sd(path.sum[t], path.sum_sq[t], run.seeds);
                            std::fprintf(paths_file, "%s,%s,%s,%g,%s,%d,%zu,%.6g,%.6g\n", run.cop.c_str(), scenario.c_str(),
                                         estimators[e].c_str(), kShareRates[rate], loss, m, t, mean, sd);
                        }
                    }
            }
    }
    std::fclose(seeds_file);
    std::fclose(paths_file);
    return 0;
}

}  // namespace strategy_experiments
