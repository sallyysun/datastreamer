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
using strategy_experiments::SeedResults;
using strategy_experiments::Simulation;

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
                    rt[t] = s.true_spearman[t];
                    for (int di = 0; di < 2; ++di) {
                        const double f = m[di].estimate(arms.estimators()), d = f - s.true_spearman[t];
                        se[di][t] = d * d;
                        const std::size_t b = (std::size_t(ei) * 2 + di) * kN + t;
                        es[b] += d * d; fsum[b] += f;
                        for (std::size_t k = 0; k < kK; ++k) ws[b * kK + k] += m[di].w[k];
                    }
                    for (int di = 0; di < 2; ++di) m[di].score_and_update_weights(arms.estimators(), s.estimated_pit_x[t], s.estimated_pit_y[t], t);
                    arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
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

struct StrategyOptions {
    bool mixdens, paths;
    const char* dump_seed;
    std::string dump_scenario;
};

// The strategies_* outputs and the per-scenario sums across seeds behind them.
class StrategyResults {
public:
    StrategyResults(const Run& run, const StrategyOptions& options);
    ~StrategyResults();
    StrategyResults(const StrategyResults&) = delete;
    StrategyResults& operator=(const StrategyResults&) = delete;

    bool ok() const { return ok_; }
    void begin_scenario(const std::string& scenario);
    void add(int seed_index, const SeedResults& result);
    void end_scenario();

private:
    // One estimator and strategy across the seeds of a scenario.
    struct StrategyTotals {
        std::vector<double> squared_error_sum;              // [t]
        std::vector<double> h2_sum_x1000;                   // [t], PATHS only
        std::vector<int> extreme_seeds_by_tick;             // [t], PATHS only
        std::vector<float> excess_x1000, forecast;          // [seed * kN + t], PATHS only
        std::vector<float> excess_after_burn_in_x1000;      // all seeds x ticks, extremes only
        int extreme_seeds = 0, extreme_seed_ticks = 0;      // extremes only
    };

    StrategyTotals& totals(std::size_t estimator, std::size_t strategy) { return totals_[estimator * strategy_count_ + strategy]; }
    void write_extremes();
    void write_paths();
    void write_jump_path();

    const std::string& cop_;
    const int seeds_;
    const bool paths_;
    const std::size_t strategy_count_ = kStrat.size();
    std::FILE* mixture_check_file_ = nullptr;
    std::FILE* seed_scores_file_ = nullptr;
    std::FILE* jump_path_file_ = nullptr;
    std::FILE* tick_paths_file_ = nullptr;
    std::FILE* seed_dump_file_ = nullptr;
    std::FILE* extremes_file_ = nullptr;
    bool ok_ = false;

    std::string scenario_;
    std::vector<thesis::Win> windows_;
    std::vector<StrategyTotals> totals_;   // [estimator * strategy_count_ + strategy]
    std::vector<double> true_spearman_;
};

StrategyResults::StrategyResults(const Run& run, const StrategyOptions& options)
    : cop_(run.cop), seeds_(run.seeds), paths_(options.paths) {
    const std::string& sfx = run.sfx;
    const char* dseed = options.dump_seed;
    auto open = [&](const std::string& name) { return std::fopen(("results/strategies_" + name + sfx + ".csv").c_str(), "w"); };
    mixture_check_file_ = options.mixdens ? open("mixcheck") : nullptr;
    if (mixture_check_file_) std::fprintf(mixture_check_file_, "copula,scenario,estimator,seed,mixloss_kll\n");
    seed_scores_file_ = paths_ ? std::fopen("/dev/null", "w") : open("seeds");
    jump_path_file_ = paths_ ? std::fopen("/dev/null", "w") : open("jumppath");
    tick_paths_file_ = (paths_ && !dseed) ? open("paths") : nullptr;
    seed_dump_file_ = dseed ? open("dump_" + options.dump_scenario + "_s" + dseed) : nullptr;
    if (seed_dump_file_) std::fprintf(seed_dump_file_, "estimator,strategy,t,forecast,rho_true\n");
    extremes_file_ = (run.tag && !paths_) ? open("extremes") : nullptr;
    if (!seed_scores_file_ || !jump_path_file_ || (options.mixdens && !mixture_check_file_) ||
        (paths_ && !dseed && !tick_paths_file_) || (dseed && !seed_dump_file_) || (run.tag && !paths_ && !extremes_file_))
        return;
    ok_ = true;
    if (extremes_file_) std::fprintf(extremes_file_, "copula,scenario,estimator,strategy,ex_mean,ex_median,ex_max,top1pct_share,seeds_extreme,seedticks_extreme\n");
    if (tick_paths_file_) std::fprintf(tick_paths_file_, "copula,scenario,estimator,strategy,t,mse,ex_mean,ex_median,ex_max,h2_mean,n_extreme,rho_true,est_mean,est_q10,est_q90\n");
    std::fprintf(seed_scores_file_, "copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(jump_path_file_, "copula,estimator,strategy,t,mse\n");
}

StrategyResults::~StrategyResults() {
    for (std::FILE* f : {seed_scores_file_, jump_path_file_, tick_paths_file_, seed_dump_file_, extremes_file_, mixture_check_file_})
        if (f) std::fclose(f);
}

void StrategyResults::begin_scenario(const std::string& scenario) {
    scenario_ = scenario;
    windows_ = windows(scenario);
    true_spearman_.assign(kN, 0.0);
    totals_.assign(3 * strategy_count_, {});
    for (auto& cell : totals_) {
        cell.squared_error_sum.assign(kN, 0.0);
        if (paths_) {
            cell.h2_sum_x1000.assign(kN, 0.0);
            cell.extreme_seeds_by_tick.assign(kN, 0);
            cell.excess_x1000.assign(std::size_t(seeds_) * kN, 0.0f);
            cell.forecast.assign(std::size_t(seeds_) * kN, 0.0f);
        }
    }
}

void StrategyResults::add(int seed_index, const SeedResults& result) {
    for (std::size_t estimator = 0; estimator < 3; ++estimator) {
        const std::string name = kEst[estimator];
        const auto& traces = result.estimators[estimator].strategies;
        std::vector<char> seed_had_extreme(strategy_count_, 0);
        for (std::size_t t = 0; t < kN; ++t) {
            if (seed_dump_file_)
                for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy)
                    std::fprintf(seed_dump_file_, "%s,%s,%zu,%.10g,%.10g\n", name.c_str(), kStrat[strategy].c_str(), t,
                                 traces[strategy].forecast[t], result.true_spearman[t]);
            for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy) {
                const auto& trace = traces[strategy];
                auto& cell = totals(estimator, strategy);
                const double forecast = trace.forecast[t];
                cell.squared_error_sum[t] += trace.squared_error[t];
                if (paths_) {
                    const std::size_t sample = std::size_t(seed_index) * kN + t;
                    cell.excess_x1000[sample] = float(1000 * trace.excess[t]);
                    cell.forecast[sample] = float(forecast);
                    true_spearman_[t] = result.true_spearman[t];
                    if (checkpoint(scenario_, t)) cell.h2_sum_x1000[t] += 1000 * trace.h2[t];
                    if (std::fabs(forecast) > 0.98) ++cell.extreme_seeds_by_tick[t];
                }
            }
            if (extremes_file_ && t >= kBurn)
                for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy) {
                    const auto& trace = traces[strategy];
                    auto& cell = totals(estimator, strategy);
                    cell.excess_after_burn_in_x1000.push_back(float(1000 * trace.excess[t]));
                    if (std::fabs(trace.forecast[t]) > 0.98) { ++cell.extreme_seed_ticks; seed_had_extreme[strategy] = 1; }
                }
        }
        if (extremes_file_)
            for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy)
                totals(estimator, strategy).extreme_seeds += seed_had_extreme[strategy];
        if (mixture_check_file_)
            std::fprintf(mixture_check_file_, "%s,%s,%s,%u,%.8f\n", cop_.c_str(), scenario_.c_str(), name.c_str(), result.seed,
                         result.estimators[estimator].mixture_log_loss);
        for (const auto& w : windows_) {
            for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy) {
                const auto& trace = traces[strategy];
                const auto [rmse, excess, h2_score] = summarize_window(w, trace.squared_error, trace.excess, trace.h2);
                std::fprintf(seed_scores_file_, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", cop_.c_str(), scenario_.c_str(), name.c_str(),
                             kStrat[strategy].c_str(), result.seed, w.name, rmse, excess, h2_score);
            }
        }
    }
}

void StrategyResults::end_scenario() {
    if (extremes_file_) write_extremes();
    if (tick_paths_file_) write_paths();
    if (scenario_ == "jump") write_jump_path();
}

void StrategyResults::write_extremes() {
    for (std::size_t estimator = 0; estimator < 3; ++estimator)
        for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy) {
            const auto& cell = totals(estimator, strategy);
            auto v = cell.excess_after_burn_in_x1000;
            double sum = 0, mx = -INFINITY;
            for (float x : v) { sum += x; mx = std::max(mx, double(x)); }
            std::sort(v.begin(), v.end());
            const double med = 0.5 * (v[(v.size() - 1) / 2] + v[v.size() / 2]);
            const std::size_t k = std::max<std::size_t>(1, v.size() / 100);
            double top = 0;
            for (std::size_t j = v.size() - k; j < v.size(); ++j) top += v[j];
            std::fprintf(extremes_file_, "%s,%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%d,%d\n", cop_.c_str(), scenario_.c_str(), kEst[estimator],
                         kStrat[strategy].c_str(), sum / v.size(), med, mx, sum != 0 ? top / sum : NAN, cell.extreme_seeds,
                         cell.extreme_seed_ticks);
        }
}

void StrategyResults::write_paths() {
    const int seeds = seeds_;
    for (std::size_t estimator = 0; estimator < 3; ++estimator)
        for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy) {
            const auto& cell = totals(estimator, strategy);
            for (std::size_t t = kBurn; t < kN; ++t) {
                std::vector<float> v(seeds);
                double sum = 0, mx = -INFINITY;
                for (int sd = 0; sd < seeds; ++sd) {
                    v[sd] = cell.excess_x1000[std::size_t(sd) * kN + t];
                    sum += v[sd];
                    mx = std::max(mx, double(v[sd]));
                }
                std::nth_element(v.begin(), v.begin() + seeds / 2, v.end());
                double med = v[seeds / 2];
                if (seeds % 2 == 0) med = 0.5 * (med + *std::max_element(v.begin(), v.begin() + seeds / 2));
                std::fprintf(tick_paths_file_, "%s,%s,%s,%s,%zu,%.8g,%.6g,%.6g,%.6g,", cop_.c_str(), scenario_.c_str(), kEst[estimator],
                             kStrat[strategy].c_str(), t, cell.squared_error_sum[t] / seeds, sum / seeds, med, mx);
                if (checkpoint(scenario_, t)) std::fprintf(tick_paths_file_, "%.6g", cell.h2_sum_x1000[t] / seeds);
                std::vector<float> fv(seeds);
                double fsum = 0;
                for (int sd = 0; sd < seeds; ++sd) { fv[sd] = cell.forecast[std::size_t(sd) * kN + t]; fsum += fv[sd]; }
                std::sort(fv.begin(), fv.end());
                auto q = [&](double p) { const double x = p * (seeds - 1); const int k = int(x); return fv[k] + (x - k) * (fv[std::min(k + 1, seeds - 1)] - fv[k]); };
                std::fprintf(tick_paths_file_, ",%d,%.6g,%.6g,%.6g,%.6g\n", cell.extreme_seeds_by_tick[t], true_spearman_[t], fsum / seeds,
                             q(0.1), q(0.9));
            }
        }
}

void StrategyResults::write_jump_path() {
    for (std::size_t estimator = 0; estimator < 3; ++estimator)
        for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy)
            for (std::size_t t = kBurn; t < kN; ++t)
                std::fprintf(jump_path_file_, "%s,%s,%s,%zu,%.8g\n", cop_.c_str(), kEst[estimator], kStrat[strategy].c_str(), t,
                             totals(estimator, strategy).squared_error_sum[t] / seeds_);
}

// The standard strategy comparison: one Simulation per scenario and seed, collated by StrategyResults.
// MIXDENS=1 scores the weighted density mixture. Its KLL log loss also checks the regret calculation.
class Experiment {
public:
    explicit Experiment(const Run& run) : run_(run) {}
    int run();

private:
    const Run& run_;
};

int Experiment::run() {
    StrategyOptions options{std::getenv("MIXDENS") != nullptr, false, std::getenv("DUMP_SEED"),
                            std::getenv("DUMP_SCEN") ? std::getenv("DUMP_SCEN") : "jump"};
    if (options.mixdens && !run_.tag) { std::fprintf(stderr, "MIXDENS needs TAG\n"); return 1; }
    options.paths = std::getenv("PATHS") != nullptr || options.dump_seed;   // per-tick paths / dump only; the other outputs are left alone
    StrategyResults results(run_, options);
    if (!results.ok()) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }

    for (const std::string sc : {"constant", "drift", "jump"}) {
        if (options.dump_seed && sc != options.dump_scenario) continue;
        results.begin_scenario(sc);
        for (int sd = 0; sd < run_.seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            if (options.dump_seed && int(seed) != std::atoi(options.dump_seed)) continue;
            results.add(sd, Simulation(run_, sc, seed, kRates, options.mixdens).run());
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
    return Experiment(run).run();
}
