#include "simulation_reports.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace strategy_experiments {

using thesis::checkpoint;
using thesis::kBurn;
using thesis::kN;
using thesis::windows;

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

ForecastComparisonReport::ForecastComparisonReport(const Run& run, const StrategyOptions& options,
                                 const std::vector<std::string>& strategy_labels)
    : cop_(run.cop), seeds_(run.seeds), paths_(options.paths), strategy_labels_(strategy_labels),
      strategy_count_(strategy_labels.size()) {
    const std::string& sfx = run.sfx;
    const char* dseed = options.dump_seed;
    auto open = [&](const std::string& name) { return open_result("strategies_" + name + sfx); };
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

ForecastComparisonReport::~ForecastComparisonReport() {
    for (std::FILE* f : {seed_scores_file_, jump_path_file_, tick_paths_file_, seed_dump_file_, extremes_file_, mixture_check_file_})
        if (f) std::fclose(f);
}

void ForecastComparisonReport::begin_scenario(const std::string& scenario) {
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

void ForecastComparisonReport::add(int seed_index, const SeedResults& result) {
    for (std::size_t estimator = 0; estimator < 3; ++estimator) {
        const std::string name = kEst[estimator];
        const auto& traces = result.estimators[estimator].strategies;
        std::vector<char> seed_had_extreme(strategy_count_, 0);
        for (std::size_t t = 0; t < kN; ++t) {
            if (seed_dump_file_)
                for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy)
                    std::fprintf(seed_dump_file_, "%s,%s,%zu,%.10g,%.10g\n", name.c_str(), strategy_labels_[strategy].c_str(), t,
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
                             strategy_labels_[strategy].c_str(), result.seed, w.name, rmse, excess, h2_score);
            }
        }
    }
}

void ForecastComparisonReport::end_scenario() {
    if (extremes_file_) write_extremes();
    if (tick_paths_file_) write_paths();
    if (scenario_ == "jump") write_jump_path();
}

void ForecastComparisonReport::write_extremes() {
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
                         strategy_labels_[strategy].c_str(), sum / v.size(), med, mx, sum != 0 ? top / sum : NAN, cell.extreme_seeds,
                         cell.extreme_seed_ticks);
        }
}

void ForecastComparisonReport::write_paths() {
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
                             strategy_labels_[strategy].c_str(), t, cell.squared_error_sum[t] / seeds, sum / seeds, med, mx);
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

void ForecastComparisonReport::write_jump_path() {
    for (std::size_t estimator = 0; estimator < 3; ++estimator)
        for (std::size_t strategy = 0; strategy < strategy_count_; ++strategy)
            for (std::size_t t = kBurn; t < kN; ++t)
                std::fprintf(jump_path_file_, "%s,%s,%s,%zu,%.8g\n", cop_.c_str(), kEst[estimator], strategy_labels_[strategy].c_str(), t,
                             totals(estimator, strategy).squared_error_sum[t] / seeds_);
}

MffWeightPathReport::MffWeightPathReport(const Run& run)
    : cop_(run.cop), seeds_(run.seeds), weights_file_(open_result("strategies_weights" + run.sfx)),
      window_file_(open_result("strategies_weights_seeds" + run.sfx)) {
    if (!ok()) return;
    std::fprintf(weights_file_, "copula,scenario,estimator,delta,t");
    for (double memory : thesis::kMem) std::fprintf(weights_file_, ",w%g", memory);
    std::fprintf(weights_file_, ",mse,rho_true,est_mean\n");
    std::fprintf(window_file_, "copula,scenario,estimator,delta,seed,window,rmse\n");
}

MffWeightPathReport::~MffWeightPathReport() {
    for (std::FILE* f : {weights_file_, window_file_})
        if (f) std::fclose(f);
}

void MffWeightPathReport::begin_scenario(const std::string& scenario) {
    scenario_ = scenario;
    windows_ = windows(scenario);
    weight_sum_.assign(3 * 2 * kN * thesis::kK, 0.0);
    squared_error_sum_.assign(3 * 2 * kN, 0.0);
    forecast_sum_.assign(3 * 2 * kN, 0.0);
    true_spearman_.assign(kN, 0.0);
}

void MffWeightPathReport::add(const SeedResults& result) {
    true_spearman_ = result.true_spearman;
    for (std::size_t estimator = 0; estimator < 3; ++estimator) {
        const auto& paths = result.estimators[estimator].weight_paths;
        delta_ = {paths[0].delta, paths[1].delta};
        for (std::size_t path = 0; path < 2; ++path) {
            const auto& trace = paths[path];
            for (std::size_t t = 0; t < kN; ++t) {
                const std::size_t b = (estimator * 2 + path) * kN + t;
                squared_error_sum_[b] += trace.squared_error[t];
                forecast_sum_[b] += trace.forecast[t];
                for (std::size_t k = 0; k < thesis::kK; ++k) weight_sum_[b * thesis::kK + k] += trace.weights[t * thesis::kK + k];
            }
            for (const auto& w : windows_) {
                double a = 0;
                for (std::size_t t = w.a; t < w.b; ++t) a += trace.squared_error[t];
                std::fprintf(window_file_, "%s,%s,%s,%g,%u,%s,%.8f\n", cop_.c_str(), scenario_.c_str(), kEst[estimator], trace.delta,
                             result.seed, w.name, std::sqrt(a / double(w.b - w.a)));
            }
        }
    }
}

void MffWeightPathReport::end_scenario() {
    for (std::size_t estimator = 0; estimator < 3; ++estimator)
        for (std::size_t path = 0; path < 2; ++path)
            for (std::size_t t = 0; t < kN; ++t) {
                const std::size_t b = (estimator * 2 + path) * kN + t;
                std::fprintf(weights_file_, "%s,%s,%s,%g,%zu", cop_.c_str(), scenario_.c_str(), kEst[estimator], delta_[path], t);
                for (std::size_t k = 0; k < thesis::kK; ++k) std::fprintf(weights_file_, ",%.6g", weight_sum_[b * thesis::kK + k] / seeds_);
                std::fprintf(weights_file_, ",%.8g,%.6g,%.6g\n", squared_error_sum_[b] / seeds_, true_spearman_[t], forecast_sum_[b] / seeds_);
            }
}

}  // namespace strategy_experiments
