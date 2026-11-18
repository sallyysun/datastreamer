// Regret against the best fixed arm and the best one- or two-switch sequences.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
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

// Compare family-specific mixture regret against the best arm and the best one- or two-switch paths.
int run_regretfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const double deltas[2] = {0.0, 1e-4};
    std::FILE* fr = open_result("regretfam_seeds" + sfx);
    if (!fr) return missing_results_folder();
    std::fprintf(fr, "gen,scenario,working,delta,seed,T,t0,loss_mix,best0,arm0,best1,best2,sw_t,sw_from,sw_to,n_nonfinite,n_clamp\n");
    const WorkingFamilies families;
    for (const std::string sc : {"constant", "drift", "jump"}) {
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            const StudySeries s = load_study_series(cop, sc, seed, kN);
            for (int wi = 0; wi < 4; ++wi) {
                const auto& inv = families.inverse[wi];
                const auto family = WorkingFamilies::family[wi];
                FixedMemoryArms arms("rmpl");
                thesis::Mff mix[2] = {thesis::Mff(family, kK, 1.0, deltas[0]), thesis::Mff(family, kK, 1.0, deltas[1])};
                std::vector<std::vector<double>> L;
                std::vector<std::array<double, 2>> mf;
                std::vector<std::size_t> tt;
                std::vector<double> l(kK);
                long nonfin = 0, nclamp = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    if (t >= thesis::kMffWarmup && arms.ready()) {
                        bool clamp = false;
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double rho = arms.estimate(k);
                            l[k] = -copula::log_density(family, s.estimated_pit_x[t], s.estimated_pit_y[t], inv(rho));
                            clamp = clamp || inv.clamped(rho);
                        }
                        nclamp += clamp;
                        nonfin += thesis::guard_mff_losses(l);
                        std::array<double, 2> row{};
                        for (int di = 0; di < 2; ++di) {
                            row[di] = mix[di].mixture_loss(l.data());
                            mix[di].update(l);
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t);
                    }
                    arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
                }
                const std::size_t T = L.size();
                const auto oracle = regret_oracle(L);
                for (int di = 0; di < 2; ++di) {
                    double cm = 0;
                    for (const auto& row : mf) cm += row[di];
                    const auto switch_tick = oracle.before_arm == oracle.after_arm
                        ? 0 : tt[std::min(oracle.switch_index, T - 1)];
                    std::fprintf(fr, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g,%ld,%ld\n", cop.c_str(), sc.c_str(), WorkingFamilies::name[wi],
                                 deltas[di], seed, T, tt.front(), cm, oracle.fixed_loss, kMem[oracle.fixed_arm],
                                 oracle.one_switch_loss, oracle.prefix_loss.back()[2], switch_tick,
                                 kMem[oracle.before_arm], kMem[oracle.after_arm], nonfin, nclamp);
                }
            }
        }
    }
    std::fclose(fr);
    return 0;
}

// Compare Gaussian mixture and point-forecast regret against static and switching oracles.
int run_regret(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    constexpr int MS = kSwitchBudgets;
    const double deltas[2] = {0.0, 1e-4};
    std::FILE* fr = open_result("regret_seeds" + sfx);
    std::FILE* fc = open_result("regret_paths" + sfx);
    if (!fr || !fc) return missing_results_folder();
    std::fprintf(fr, "copula,scenario,estimator,delta,seed,T,t0,loss_mix,loss_point,best0,arm0,best1,best2,sw_t,sw_from,sw_to\n");
    std::fprintf(fc, "copula,scenario,estimator,delta,loss,m,t,mean,sd\n");
    for (const std::string sc : {"constant", "drift", "jump"}) {
        // path sums over seeds: [ei][di][loss 0 mix / 1 point][m][t]
        const std::size_t P = 3 * 2 * 2 * MS * kN;
        std::vector<double> s1(P, 0.0), s2(P, 0.0);
        auto pi = [&](int ei, int di, int l, int m, std::size_t t) { return ((((std::size_t(ei) * 2 + di) * 2 + l) * MS + m) * kN) + t; };
        std::vector<char> scored(kN, 0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            const StudySeries s = load_study_series(cop, sc, seed, kN);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                FixedMemoryArms arms(e);
                thesis::Mff mix[2] = {thesis::Mff(copula::Family::Gaussian, kK, 1.0, deltas[0]),
                                      thesis::Mff(copula::Family::Gaussian, kK, 1.0, deltas[1])};
                std::vector<std::vector<double>> L;          // [scored tick][k] guarded arm loss
                std::vector<std::array<double, 4>> mf;       // mix / point loss for delta 0, 1e-4
                std::vector<std::size_t> tt;
                std::vector<double> f(kK), l(kK);
                for (std::size_t t = 0; t < kN; ++t) {
                    arms.forecasts(f.data());
                    if (t >= thesis::kMffWarmup && arms.ready()) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double par = copula::param_from_spearman(copula::Family::Gaussian, f[k]);
                            l[k] = -copula::log_density(copula::Family::Gaussian, s.estimated_pit_x[t], s.estimated_pit_y[t], par);
                        }
                        thesis::guard_mff_losses(l);
                        std::array<double, 4> row{};
                        for (int di = 0; di < 2; ++di) {
                            row[2 * di] = mix[di].mixture_loss(l.data());
                            const double r = mix[di].forecast(f.data());
                            row[2 * di + 1] = -copula::log_density(copula::Family::Gaussian, s.estimated_pit_x[t], s.estimated_pit_y[t],
                                                                   copula::param_from_spearman(copula::Family::Gaussian, r));
                            mix[di].update(l);
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t); scored[t] = 1;
                    }
                    arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
                }
                const std::size_t T = L.size();
                const auto oracle = regret_oracle(L);
                for (int di = 0; di < 2; ++di) {
                    double cm = 0, cp = 0;
                    for (std::size_t i = 0; i < T; ++i) {
                        cm += mf[i][2 * di]; cp += mf[i][2 * di + 1];
                        for (int m = 0; m < MS; ++m) {
                            const double rm = cm - oracle.prefix_loss[i][m], rp = cp - oracle.prefix_loss[i][m];
                            s1[pi(ei, di, 0, m, tt[i])] += rm; s2[pi(ei, di, 0, m, tt[i])] += rm * rm;
                            s1[pi(ei, di, 1, m, tt[i])] += rp; s2[pi(ei, di, 1, m, tt[i])] += rp * rp;
                        }
                    }
                    const auto switch_tick = oracle.before_arm == oracle.after_arm
                        ? 0 : tt[std::min(oracle.switch_index, T - 1)];
                    std::fprintf(fr, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g\n",
                                 cop.c_str(), sc.c_str(), e.c_str(), deltas[di], seed, T, tt.front(), cm, cp,
                                 oracle.fixed_loss, kMem[oracle.fixed_arm], oracle.one_switch_loss,
                                 oracle.prefix_loss.back()[2], switch_tick, kMem[oracle.before_arm], kMem[oracle.after_arm]);
                }
            }
        }
        for (int ei = 0; ei < 3; ++ei)
            for (int di = 0; di < 2; ++di)
                for (int lo = 0; lo < 2; ++lo)
                    for (int m = 0; m < MS; ++m)
                        for (std::size_t t = 0; t < kN; ++t) {
                            if (!scored[t] || (t % 10 != 9 && t != kN - 1)) continue;
                            const auto [mean, sd] = seed_mean_sd(s1[pi(ei, di, lo, m, t)], s2[pi(ei, di, lo, m, t)], seeds);
                            std::fprintf(fc, "%s,%s,%s,%g,%s,%d,%zu,%.6g,%.6g\n", cop.c_str(), sc.c_str(), kEst[ei], deltas[di],
                                         lo ? "point" : "mix", m, t, mean, sd);
                        }
    }
    std::fclose(fr); std::fclose(fc);
    return 0;
}

}  // namespace strategy_experiments
