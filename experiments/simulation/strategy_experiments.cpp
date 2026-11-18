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
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

Stream make_stream(const std::string& family, const std::string& scenario, unsigned seed) {
    Stream stream{load_study_series(family, scenario, seed, thesis::kN), {}};
    const auto generating_family = thesis::family_of(family);
    stream.par.reserve(stream.target.size());
    for (double target : stream.target) stream.par.push_back(copula::param_from_spearman(generating_family, target));
    return stream;
}

WindowScores summarize_window(const thesis::Win& window, const std::vector<double>& squared_error,
                              const std::vector<double>& excess, const std::vector<double>& hellinger) {
    const bool dense = std::string_view(window.name) == "rec_0_100";
    double se_sum = 0, excess_sum = 0, h2_sum = 0;
    std::size_t h2_count = 0;
    for (std::size_t t = window.a; t < window.b; ++t) {
        se_sum += squared_error[t];
        excess_sum += excess[t];
        if (!std::isnan(hellinger[t]) && (dense || (t - thesis::kBurn) % 50 == 0)) {
            h2_sum += hellinger[t];
            ++h2_count;
        }
    }
    const double n = double(window.b - window.a);
    return {std::sqrt(se_sum / n), 1000 * excess_sum / n, h2_count ? 1000 * h2_sum / h2_count : NAN};
}

}  // namespace strategy_experiments

namespace {
using thesis::family_of; using thesis::gauss_logc; using thesis::H2Table; using thesis::Mff;
using thesis::windows; using thesis::checkpoint;
using thesis::kN; using thesis::kBurn; using thesis::kK; using thesis::kMem;
using strategy_experiments::FixedMemoryArms;
using strategy_experiments::Run;
using strategy_experiments::Stream;
using strategy_experiments::kEst;
using strategy_experiments::make_stream;
using strategy_experiments::summarize_window;

// Strategies: five fixed arms, MFF, then one VFF per rate. Default rates 0.02 / 0.1 named vff_slow / vff_fast (the
// original outputs); VFF_RATES=a,b,... names them vff_<a>, ... (sensitivity runs; write under TAG=...).
std::vector<std::string> kStrat = thesis::strategy_names();
std::vector<double> kRates = thesis::vff_rates();

int run_validate(const Run& run) {
    const std::string& cop = run.cop;
    H2Table& h2 = run.h2;
    const double par = copula::param_from_spearman(h2.F, 0.6 * 0.97);
    for (double r : {0.123, 0.4567, 0.58, 0.7891, 0.9})
        std::printf("VALIDATE H2 %s rho_hat %.4f: table %.6e direct %.6e\n", cop.c_str(), r, h2(par, r), h2.direct(par, r));
    {   // mixture H^2: closed-form grid vs a dmetric::Table of the same mixture through copula::log_density
        const double rs[5] = {0.2, 0.45, 0.6, 0.75, 0.9}, ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15}, w1[5] = {0, 0, 1, 0, 0};
        const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], a, b)); return m; });
        std::printf("VALIDATE H2mix %s 5-comp: grid %.6e dmetric %.6e | 1-comp (rho 0.6): grid %.6e direct %.6e\n", cop.c_str(),
                    h2.mix(par, rs, ws, 5), dmetric::hellinger2(h2.g, truth, mt), h2.mix(par, rs, w1, 5), h2.direct(par, 0.6));
        const double lm = H2Table::mix_logc(rs, ws, 5, 0.3, 0.8);
        double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], 0.3, 0.8));
        std::printf("VALIDATE mix_logc: %.10f vs %.10f\n", lm, std::log(m));
    }
    return 0;
}

// MFF weight paths: delta = 1e-4 (the main MFF) and plain Hedge delta = 0, same arms
int run_weights(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    std::FILE* fw = std::fopen(("results/strategies_weights" + sfx + ".csv").c_str(), "w");
    std::FILE* fc = std::fopen(("results/strategies_weights_seeds" + sfx + ".csv").c_str(), "w");
    if (!fw || !fc) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fw, "copula,scenario,estimator,delta,t");
    for (double memory : kMem) std::fprintf(fw, ",w%g", memory);
    std::fprintf(fw, ",mse,rho_true,est_mean\n");
    std::fprintf(fc, "copula,scenario,estimator,delta,seed,window,rmse\n");
    const double deltas[2] = {1e-4, 0.0};
    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        // [ei][di][t][k]: summed weights held before tick t; [ei][di][t]: summed squared error and forecast
        std::vector<double> ws(3 * 2 * kN * kK), es(3 * 2 * kN), fsum(3 * 2 * kN), rt(kN);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            const Stream s = make_stream(cop, sc, seed);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                FixedMemoryArms arms(e);
                Mff m[2];
                for (int di = 0; di < 2; ++di) m[di].delta = deltas[di];
                std::vector<std::vector<double>> se(2, std::vector<double>(kN));
                for (std::size_t t = 0; t < kN; ++t) {
                    rt[t] = s.rho[t];
                    for (int di = 0; di < 2; ++di) {
                        const double f = m[di].estimate(arms.estimators()), d = f - s.rho[t];
                        se[di][t] = d * d;
                        const std::size_t b = (std::size_t(ei) * 2 + di) * kN + t;
                        es[b] += d * d; fsum[b] += f;
                        for (std::size_t k = 0; k < kK; ++k) ws[b * kK + k] += m[di].w[k];
                    }
                    for (int di = 0; di < 2; ++di) m[di].score(arms.estimators(), s.u[t], s.v[t], t);
                    arms.update(s.x[t], s.y[t], s.u[t], s.v[t]);
                }
                for (int di = 0; di < 2; ++di)
                    for (const auto& w : wins) {
                        double a = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) a += se[di][t];
                        std::fprintf(fc, "%s,%s,%s,%g,%u,%s,%.8f\n", cop.c_str(), sc.c_str(), e.c_str(), deltas[di], seed, w.name,
                                     std::sqrt(a / double(w.b - w.a)));
                    }
            }
        }
        for (int ei = 0; ei < 3; ++ei)
            for (int di = 0; di < 2; ++di)
                for (std::size_t t = 0; t < kN; ++t) {
                    const std::size_t b = (std::size_t(ei) * 2 + di) * kN + t;
                    std::fprintf(fw, "%s,%s,%s,%g,%zu", cop.c_str(), sc.c_str(), kEst[ei], deltas[di], t);
                    for (std::size_t k = 0; k < kK; ++k) std::fprintf(fw, ",%.6g", ws[b * kK + k] / seeds);
                    std::fprintf(fw, ",%.8g,%.6g,%.6g\n", es[b] / seeds, rt[t], fsum[b] / seeds);
                }
    }
    std::fclose(fw); std::fclose(fc);
    return 0;
}

// MIXDENS=1 scores the weighted density mixture. Its KLL log loss also checks the regret calculation.
int run_strategies(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const bool mixdens = std::getenv("MIXDENS") != nullptr;
    if (mixdens && !tag) { std::fprintf(stderr, "MIXDENS needs TAG\n"); return 1; }
    std::FILE* fmc = mixdens ? std::fopen(("results/strategies_mixcheck" + sfx + ".csv").c_str(), "w") : nullptr;
    if (fmc) std::fprintf(fmc, "copula,scenario,estimator,seed,mixloss_kll\n");
    const char* dseed = std::getenv("DUMP_SEED");
    const std::string dscen = std::getenv("DUMP_SCEN") ? std::getenv("DUMP_SCEN") : "jump";
    const bool paths = std::getenv("PATHS") != nullptr || dseed;   // per-tick paths / dump only; the other outputs are left alone
    std::FILE* fs = std::fopen(paths ? "/dev/null" : ("results/strategies_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fj = std::fopen(paths ? "/dev/null" : ("results/strategies_jumppath" + sfx + ".csv").c_str(), "w");
    std::FILE* fp = (paths && !dseed) ? std::fopen(("results/strategies_paths" + sfx + ".csv").c_str(), "w") : nullptr;
    std::FILE* fdump = dseed ? std::fopen(("results/strategies_dump_" + dscen + "_s" + dseed + sfx + ".csv").c_str(), "w") : nullptr;
    if (fdump) std::fprintf(fdump, "estimator,strategy,t,forecast,rho_true\n");
    std::FILE* fx = (tag && !paths) ? std::fopen(("results/strategies_extremes" + sfx + ".csv").c_str(), "w") : nullptr;
    if (!fs || !fj || (paths && !dseed && !fp) || (dseed && !fdump) || (tag && !paths && !fx)) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    if (fx) std::fprintf(fx, "copula,scenario,estimator,strategy,ex_mean,ex_median,ex_max,top1pct_share,seeds_extreme,seedticks_extreme\n");
    if (fp) std::fprintf(fp, "copula,scenario,estimator,strategy,t,mse,ex_mean,ex_median,ex_max,h2_mean,n_extreme,rho_true,est_mean,est_q10,est_q90\n");
    std::fprintf(fs, "copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fj, "copula,estimator,strategy,t,mse\n");
    const copula::Family F = family_of(cop);
    const std::size_t S = kStrat.size();


    for (const std::string sc : {"constant", "drift", "jump"}) {
        if (dseed && sc != dscen) continue;
        const auto wins = windows(sc);
        std::vector<std::vector<std::vector<double>>> jsum(3, std::vector<std::vector<double>>(S, std::vector<double>(kN)));
        auto at = [&](int ei, std::size_t i, int sd, std::size_t t) { return ((std::size_t(ei) * S + i) * seeds + sd) * kN + t; };
        std::vector<float> exs(paths ? 3 * S * seeds * kN : 0), fcs(paths ? 3 * S * seeds * kN : 0);   // excess, forecasts
        std::vector<double> rho_true(kN);
        std::vector<double> h2s(paths ? 3 * S * kN : 0);
        std::vector<int> nex(paths ? 3 * S * kN : 0);
        std::vector<std::vector<float>> xall(fx ? 3 * S : 0);          // whole-window excess, all seeds x ticks (x 1000)
        std::vector<int> xseeds(fx ? 3 * S : 0), xticks(fx ? 3 * S : 0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            if (dseed && int(seed) != std::atoi(dseed)) continue;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) {
                const double l = copula::log_density(F, s.pu[t], s.pv[t], s.par[t]);
                lc[t] = l;
            }
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                thesis::EstimatorStrategies strategies(e, kRates);
                const auto& weights = strategies.mff_weights();
                const auto mixture_weights = mixdens ? std::span<const double>(weights) : std::span<const double>{};
                std::vector<char> seed_ext(S, 0);
                std::vector<std::vector<double>> se(S, std::vector<double>(kN)), ex(S, std::vector<double>(kN)), hh(S, std::vector<double>(kN, NAN));
                std::vector<double> f(S);
                double mixkll = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    strategies.forecasts(f);
                    if (mixdens && strategies.mff_ready(t))
                        mixkll -= H2Table::mix_logc(f.data(), weights.data(), kK, s.u[t], s.v[t]);
                    if (fdump)
                        for (std::size_t i = 0; i < S; ++i)
                            std::fprintf(fdump, "%s,%s,%zu,%.10g,%.10g\n", e.c_str(), kStrat[i].c_str(), t, f[i], s.rho[t]);
                    const thesis::ScoringObservation truth{s.rho[t], s.pu[t], s.pv[t], s.par[t], lc[t], checkpoint(sc, t)};
                    for (std::size_t i = 0; i < S; ++i) {
                        const auto score = thesis::score_forecast(h2, truth, f, i, mixture_weights);
                        se[i][t] = score.squared_error;
                        ex[i][t] = score.excess;
                        hh[i][t] = score.h2;
                        jsum[ei][i][t] += score.squared_error;
                        if (paths) {
                            exs[at(ei, i, sd, t)] = float(1000 * ex[i][t]);
                            fcs[at(ei, i, sd, t)] = float(f[i]);
                            rho_true[t] = s.rho[t];
                            if (truth.hellinger_checkpoint) h2s[(ei * S + i) * kN + t] += 1000 * score.h2;
                            if (std::fabs(f[i]) > 0.98) ++nex[(ei * S + i) * kN + t];
                        }
                    }
                    strategies.update(s.x[t], s.y[t], s.u[t], s.v[t], t);
                    if (fx && t >= kBurn)
                        for (std::size_t i = 0; i < S; ++i) {
                            xall[ei * S + i].push_back(float(1000 * ex[i][t]));
                            if (std::fabs(f[i]) > 0.98) { ++xticks[ei * S + i]; seed_ext[i] = 1; }
                        }
                }
                if (fx) for (std::size_t i = 0; i < S; ++i) xseeds[ei * S + i] += seed_ext[i];
                if (fmc) std::fprintf(fmc, "%s,%s,%s,%u,%.8f\n", cop.c_str(), sc.c_str(), e.c_str(), seed, mixkll);
                for (const auto& w : wins) {
                    for (std::size_t i = 0; i < S; ++i) {
                        const auto [rmse, excess, h2_score] = summarize_window(w, se[i], ex[i], hh[i]);
                        std::fprintf(fs, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), e.c_str(), kStrat[i].c_str(),
                                     seed, w.name, rmse, excess, h2_score);
                    }
                }
            }
        }
        if (fx)
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i) {
                    auto v = xall[ei * S + i];
                    double sum = 0, mx = -INFINITY;
                    for (float x : v) { sum += x; mx = std::max(mx, double(x)); }
                    std::sort(v.begin(), v.end());
                    const double med = 0.5 * (v[(v.size() - 1) / 2] + v[v.size() / 2]);
                    const std::size_t k = std::max<std::size_t>(1, v.size() / 100);
                    double top = 0;
                    for (std::size_t j = v.size() - k; j < v.size(); ++j) top += v[j];
                    std::fprintf(fx, "%s,%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%d,%d\n", cop.c_str(), sc.c_str(), kEst[ei], kStrat[i].c_str(),
                                 sum / v.size(), med, mx, sum != 0 ? top / sum : NAN, xseeds[ei * S + i], xticks[ei * S + i]);
                }
        if (fp)
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i)
                    for (std::size_t t = kBurn; t < kN; ++t) {
                        std::vector<float> v(seeds);
                        double sum = 0, mx = -INFINITY;
                        for (int sd = 0; sd < seeds; ++sd) { v[sd] = exs[at(ei, i, sd, t)]; sum += v[sd]; mx = std::max(mx, double(v[sd])); }
                        std::nth_element(v.begin(), v.begin() + seeds / 2, v.end());
                        double med = v[seeds / 2];
                        if (seeds % 2 == 0) med = 0.5 * (med + *std::max_element(v.begin(), v.begin() + seeds / 2));
                        const bool cp = checkpoint(sc, t);
                        std::fprintf(fp, "%s,%s,%s,%s,%zu,%.8g,%.6g,%.6g,%.6g,", cop.c_str(), sc.c_str(), kEst[ei], kStrat[i].c_str(), t,
                                     jsum[ei][i][t] / seeds, sum / seeds, med, mx);
                        if (cp) std::fprintf(fp, "%.6g", h2s[(ei * S + i) * kN + t] / seeds);
                        std::vector<float> fv(seeds);
                        double fs_ = 0;
                        for (int sd = 0; sd < seeds; ++sd) { fv[sd] = fcs[at(ei, i, sd, t)]; fs_ += fv[sd]; }
                        std::sort(fv.begin(), fv.end());
                        auto q = [&](double p) { const double x = p * (seeds - 1); const int k = int(x); return fv[k] + (x - k) * (fv[std::min(k + 1, seeds - 1)] - fv[k]); };
                        std::fprintf(fp, ",%d,%.6g,%.6g,%.6g,%.6g\n", nex[(ei * S + i) * kN + t], rho_true[t], fs_ / seeds, q(0.1), q(0.9));
                    }
        if (sc == "jump")
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i)
                    for (std::size_t t = kBurn; t < kN; ++t)
                        std::fprintf(fj, "%s,%s,%s,%zu,%.8g\n", cop.c_str(), kEst[ei], kStrat[i].c_str(), t, jsum[ei][i][t] / seeds);
    }
    std::fclose(fs); std::fclose(fj);
    if (fp) std::fclose(fp);
    if (fdump) std::fclose(fdump);
    if (fx) std::fclose(fx);
    if (fmc) std::fclose(fmc);
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
        std::string rs(r);
        for (std::size_t a = 0, b; a <= rs.size(); a = b + 1) {
            b = rs.find(',', a); if (b == std::string::npos) b = rs.size();
            if (b > a) { kRates.push_back(std::atof(rs.substr(a, b - a).c_str())); kStrat.push_back("vff_" + rs.substr(a, b - a)); }
        }
    }
    H2Table h2{.F = family_of(cop)};
    const Run run{seeds, cop, sfx, tag, h2};
    if (std::getenv("VALIDATE")) return run_validate(run);
    if (std::getenv("FAMSWITCH")) return strategy_experiments::run_famswitch(run);
    if (std::getenv("SCORESFAM")) return strategy_experiments::run_scoresfam(run);
    if (std::getenv("REGRETFAM")) return strategy_experiments::run_regretfam(run);
    if (std::getenv("REGRET")) return strategy_experiments::run_regret(run);
    if (std::getenv("WEIGHTS")) return run_weights(run);
    return run_strategies(run);
}
