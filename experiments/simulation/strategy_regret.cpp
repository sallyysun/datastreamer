// Regret against fixed and switching oracle forecasts.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "../../src/special_functions.hpp"
#include "regret_oracle.hpp"
#include "strategy_runs.hpp"

namespace strategy_experiments {

using thesis::kK;
using thesis::kMem;
using thesis::kN;

// Compare family-specific mixture regret against the best arm and the best one- or two-switch paths.
int run_regretfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const double deltas[2] = {0.0, 1e-4};
    std::FILE* fr = std::fopen(("results/regretfam_seeds" + sfx + ".csv").c_str(), "w");
    if (!fr) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fr, "gen,scenario,working,delta,seed,T,t0,loss_mix,best0,arm0,best1,best2,sw_t,sw_from,sw_to,n_nonfinite,n_clamp\n");
    const FamilyInverse inverses[4] = {FamilyInverse(WF[0]), FamilyInverse(WF[1]), FamilyInverse(WF[2]), FamilyInverse(WF[3])};
    for (const std::string sc : {"constant", "drift", "jump"}) {
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            const Stream s = make_stream(cop, sc, seed);
            for (int wi = 0; wi < 4; ++wi) {
                const auto& inv = inverses[wi];
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm("rmpl", n));
                std::vector<std::vector<double>> w(2, std::vector<double>(kK, 1.0 / kK));
                std::vector<std::vector<double>> L;
                std::vector<std::array<double, 2>> mf;
                std::vector<std::size_t> tt;
                std::vector<double> l(kK), lw(kK);
                long nonfin = 0, nclamp = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    bool ready = t >= 30;
                    for (const auto& a : arms) ready = ready && a->ready();
                    if (ready) {
                        bool clamp = false;
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double rho = arms[k]->estimate();
                            l[k] = -copula::log_density(WF[wi], s.u[t], s.v[t], inv(rho));
                            clamp = clamp || inv.clamped(rho);
                        }
                        nclamp += clamp;
                        nonfin += thesis::guard_mff_losses(l);
                        std::array<double, 2> row{};
                        for (int di = 0; di < 2; ++di) {
                            double lo = INFINITY, mix = 0;
                            for (std::size_t k = 0; k < kK; ++k) lo = std::min(lo, l[k]);
                            for (std::size_t k = 0; k < kK; ++k) mix += w[di][k] * std::exp(-(l[k] - lo));
                            row[di] = lo - std::log(mix);
                            thesis::apply_fixed_share(w[di], l, lw, 1.0, deltas[di]);
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t);
                    }
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) a->update(zx, zy);
                }
                const std::size_t T = L.size();
                const auto oracle = regret_oracle(L);
                for (int di = 0; di < 2; ++di) {
                    double cm = 0;
                    for (const auto& row : mf) cm += row[di];
                    const auto switch_tick = oracle.before_arm == oracle.after_arm
                        ? 0 : tt[std::min(oracle.switch_index, T - 1)];
                    std::fprintf(fr, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g,%ld,%ld\n", cop.c_str(), sc.c_str(), WN[wi],
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
    std::FILE* fr = std::fopen(("results/regret_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fc = std::fopen(("results/regret_paths" + sfx + ".csv").c_str(), "w");
    if (!fr || !fc) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
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
            const Stream s = make_stream(cop, sc, seed);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                const bool raw = e != "rmpl";
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(e, n));
                std::vector<std::vector<double>> w(2, std::vector<double>(kK, 1.0 / kK));
                std::vector<std::vector<double>> L;          // [scored tick][k] guarded arm loss
                std::vector<std::array<double, 4>> mf;       // mix / point loss for delta 0, 1e-4
                std::vector<std::size_t> tt;
                std::vector<double> f(kK), l(kK), lw(kK);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();
                    bool ready = t >= 30;
                    for (const auto& a : arms) ready = ready && a->ready();
                    if (ready) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double par = copula::param_from_spearman(copula::Family::Gaussian, f[k]);
                            l[k] = -copula::log_density(copula::Family::Gaussian, s.u[t], s.v[t], par);
                        }
                        thesis::guard_mff_losses(l);
                        std::array<double, 4> row{};
                        for (int di = 0; di < 2; ++di) {
                            double lo = INFINITY, r = 0;
                            for (std::size_t k = 0; k < kK; ++k) { lo = std::min(lo, l[k]); r += w[di][k] * f[k]; }
                            double mix = 0;
                            for (std::size_t k = 0; k < kK; ++k) mix += w[di][k] * std::exp(-(l[k] - lo));
                            row[2 * di] = lo - std::log(mix);
                            r = std::clamp(r, -1.0, 1.0);
                            row[2 * di + 1] = -copula::log_density(copula::Family::Gaussian, s.u[t], s.v[t],
                                                                   copula::param_from_spearman(copula::Family::Gaussian, r));
                            thesis::apply_fixed_share(w[di], l, lw, 1.0, deltas[di]);
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t); scored[t] = 1;
                    }
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
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
                            const double mu = s1[pi(ei, di, lo, m, t)] / seeds;
                            const double var = std::max(0.0, s2[pi(ei, di, lo, m, t)] / seeds - mu * mu) * seeds / (seeds - 1.0);
                            std::fprintf(fc, "%s,%s,%s,%g,%s,%d,%zu,%.6g,%.6g\n", cop.c_str(), sc.c_str(), kEst[ei], deltas[di],
                                         lo ? "point" : "mix", m, t, mu, std::sqrt(var));
                        }
    }
    std::fclose(fr); std::fclose(fc);
    return 0;
}

}  // namespace strategy_experiments
