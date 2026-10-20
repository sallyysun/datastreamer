// Is a detector blind, or just badly thresholded?
//
// A hit rate of 1/20 on the `shape` scenario is consistent with two very
// different worlds: the monitored statistic does not move at all, or it moves
// and h discards it. AUC separates them, because it is threshold-free.
//
// AUC = P(a post-change block's statistic > a pre-change block's), Mann-Whitney
// with midranks. 0.5 = no information.
//
// ⚠️ CUSUM / GLR / SR are accumulators: their statistics can drift upward under
// the null purely because they are sums, so a raw post-vs-pre AUC measures drift
// as much as signal. The correction is a matched null run -- same seed, same
// ConstantDependence, Gaussian copula throughout, same windows. Because
// SwitchingCopula delegates to GaussianCopula before the switch, ticks 0..2499
// are BIT-IDENTICAL between the two runs: the pre-change sample is literally
// shared, so the null AUC isolates accumulator drift alone and
//     AUC_corrected = AUC_shape - AUC_null + 0.5.
//
// Usage: ./probe_auc [n_seeds] [base_seed]      (defaults 20, 42)
// Writes results/auc_shape.csv and results/auc_blocks.csv.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../src/estimators/dependence/change_detector.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000;       // same N as multiseed.cpp
constexpr std::size_t kSwitch = 2500;  // where make_copula("gauss2t") switches
constexpr std::size_t kBurn = 500;     // same burn-in as multiseed.cpp
constexpr std::size_t kBlock = 20;     // every detector here blocks at 20 ticks
constexpr std::size_t kEarlyPost = 3500;

const std::vector<std::string>& detector_keys() {
    static const std::vector<std::string> keys = {
        "kalman_cusum", "kalman_glr", "kalman_sr", "kalman_ewma_chart", "kalman_bocpd",
        "hermite_cusum2", "gaussian_mle_cusum", "tail_cusum",
        "kalman_var_chart", "kalman_var_chart_slow", "kalman_var_cusum"};
    return keys;
}

struct Run {
    std::vector<std::vector<double>> stat;  // [estimator][block sample]
    std::vector<std::size_t> tick;          // tick of each block sample
    std::vector<int> first_alarm_after;     // -1 = never
};

Run run_one(const std::string& family, const std::vector<std::string>& keys, unsigned seed) {
    ConstantDependence dependence(0.6);
    auto copula = dgp::make_copula(family);
    CopulaSource source(dependence, *copula, seed);

    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    for (const auto& k : keys) ests.push_back(make_estimator(k));
    std::vector<ChangeDetector*> dets(keys.size(), nullptr);
    for (std::size_t e = 0; e < keys.size(); ++e) dets[e] = dynamic_cast<ChangeDetector*>(ests[e].get());

    Run r;
    r.stat.assign(keys.size(), {});
    r.first_alarm_after.assign(keys.size(), -1);

    for (std::size_t t = 0; t < kN; ++t) {
        const CopulaSource::Sample s = source.next();
        const bool sample_here = ((t + 1) % kBlock == 0);
        if (sample_here) r.tick.push_back(t);
        for (std::size_t e = 0; e < keys.size(); ++e) {
            ests[e]->update(s.x, s.y);
            if (dets[e] && dets[e]->alarmed_this_block() && t >= kSwitch &&
                r.first_alarm_after[e] < 0) {
                r.first_alarm_after[e] = static_cast<int>(t);
            }
            if (sample_here) r.stat[e].push_back(dets[e] ? dets[e]->monitor_statistic() : 0.0);
        }
    }
    return r;
}

// Mann-Whitney U / (n_a n_b), midranks so the CUSUM floor's exact-zero ties do
// not inflate it.
double auc(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t na = a.size(), nb = b.size();
    if (na == 0 || nb == 0) return std::nan("");
    std::vector<double> all;
    all.reserve(na + nb);
    all.insert(all.end(), a.begin(), a.end());
    all.insert(all.end(), b.begin(), b.end());

    std::vector<std::size_t> order(all.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&all](std::size_t i, std::size_t j) { return all[i] < all[j]; });

    std::vector<double> rank(all.size());
    std::size_t i = 0;
    while (i < all.size()) {
        std::size_t j = i;
        while (j + 1 < all.size() && all[order[j + 1]] == all[order[i]]) ++j;
        const double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (std::size_t k = i; k <= j; ++k) rank[order[k]] = avg;
        i = j + 1;
    }
    double sum_b = 0.0;   // b is the "positive" (post-change) sample
    for (std::size_t k = 0; k < nb; ++k) sum_b += rank[na + k];
    const double u = sum_b - static_cast<double>(nb) * (static_cast<double>(nb) + 1.0) / 2.0;
    return u / (static_cast<double>(na) * static_cast<double>(nb));
}

struct Stats { double mean = 0.0, se = 0.0; std::size_t n = 0; };

Stats mean_stderr(const std::vector<double>& v) {
    Stats s;
    for (double x : v) if (!std::isnan(x)) ++s.n;
    if (s.n == 0) return s;
    double sum = 0.0;
    for (double x : v) if (!std::isnan(x)) sum += x;
    s.mean = sum / static_cast<double>(s.n);
    if (s.n < 2) return s;
    double var = 0.0;
    for (double x : v) if (!std::isnan(x)) var += (x - s.mean) * (x - s.mean);
    var /= static_cast<double>(s.n - 1);
    s.se = std::sqrt(var / static_cast<double>(s.n));
    return s;
}

void split(const Run& r, std::size_t e, std::size_t post_end,
           std::vector<double>& pre, std::vector<double>& post) {
    for (std::size_t i = 0; i < r.tick.size(); ++i) {
        const double v = r.stat[e][i];
        if (std::isnan(v)) continue;
        if (r.tick[i] >= kBurn && r.tick[i] < kSwitch) pre.push_back(v);
        else if (r.tick[i] >= kSwitch && r.tick[i] < post_end) post.push_back(v);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!check_registry()) return 1;
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 20;
    const unsigned base_seed = (argc > 2) ? static_cast<unsigned>(std::atoi(argv[2])) : 42;
    const std::vector<std::string>& keys = detector_keys();
    const std::size_t E = keys.size();

    std::vector<std::vector<double>> auc_shape(E), auc_null(E), auc_shape_early(E), auc_null_early(E);
    std::vector<int> hits(E, 0);
    std::vector<int> hits_null(E, 0);
    // Pooled: every seed's blocks thrown into one pair of samples.
    std::vector<std::vector<double>> pool_pre(E), pool_post(E), pool_post_null(E);

    std::ofstream blocks("results/auc_blocks.csv");
    blocks << "seed,run,estimator,tick,phase,stat\n";

    for (int s = 0; s < n_seeds; ++s) {
        const unsigned seed = base_seed + static_cast<unsigned>(s);
        const Run shape = run_one("gauss2t", keys, seed);
        const Run null_ = run_one("gaussian", keys, seed);

        for (std::size_t e = 0; e < E; ++e) {
            if (shape.first_alarm_after[e] >= 0) ++hits[e];
            if (null_.first_alarm_after[e] >= 0) ++hits_null[e];

            std::vector<double> pre_s, post_s, pre_n, post_n;
            split(shape, e, kN, pre_s, post_s);
            split(null_, e, kN, pre_n, post_n);
            auc_shape[e].push_back(auc(pre_s, post_s));
            auc_null[e].push_back(auc(pre_n, post_n));

            std::vector<double> pre_s2, post_s2, pre_n2, post_n2;
            split(shape, e, kEarlyPost, pre_s2, post_s2);
            split(null_, e, kEarlyPost, pre_n2, post_n2);
            auc_shape_early[e].push_back(auc(pre_s2, post_s2));
            auc_null_early[e].push_back(auc(pre_n2, post_n2));

            pool_pre[e].insert(pool_pre[e].end(), pre_s.begin(), pre_s.end());
            pool_post[e].insert(pool_post[e].end(), post_s.begin(), post_s.end());
            pool_post_null[e].insert(pool_post_null[e].end(), post_n.begin(), post_n.end());

            for (std::size_t i = 0; i < shape.tick.size(); ++i) {
                if (shape.tick[i] < kBurn) continue;
                blocks << seed << ",shape," << keys[e] << "," << shape.tick[i] << ","
                       << (shape.tick[i] < kSwitch ? "pre" : "post") << "," << shape.stat[e][i] << "\n";
                blocks << seed << ",null," << keys[e] << "," << null_.tick[i] << ","
                       << (null_.tick[i] < kSwitch ? "pre" : "post") << "," << null_.stat[e][i] << "\n";
            }
        }
        std::fprintf(stderr, "seed %u done\n", seed);
    }

    std::printf("n_seeds=%d base_seed=%u  N=%zu switch=%zu burn=%zu\n", n_seeds, base_seed, kN, kSwitch, kBurn);
    std::printf("pre window = ticks [%zu,%zu)   post window = ticks [%zu,%zu)\n\n", kBurn, kSwitch, kSwitch, kN);
    std::printf("%-20s %6s %16s %16s %16s %16s\n", "detector", "hits",
                "AUC shape", "AUC null(ctrl)", "AUC corrected", "AUC corr (early)");

    std::ofstream out("results/auc_shape.csv");
    out << "estimator,hits,n_seeds,auc_shape_mean,auc_shape_se,auc_null_mean,auc_null_se,"
           "auc_corrected,auc_corrected_se,auc_corrected_early,auc_corrected_early_se,"
           "auc_pooled_shape,auc_pooled_null,auc_pooled_corrected,null_hits\n";

    for (std::size_t e = 0; e < E; ++e) {
        const Stats ss = mean_stderr(auc_shape[e]);
        const Stats ns = mean_stderr(auc_null[e]);
        // Paired per seed: the two runs share their pre-sample bit for bit, so
        // the difference is far tighter than the two marginal SEs suggest.
        std::vector<double> diff, diff_early;
        for (std::size_t i = 0; i < auc_shape[e].size(); ++i) {
            diff.push_back(auc_shape[e][i] - auc_null[e][i] + 0.5);
            diff_early.push_back(auc_shape_early[e][i] - auc_null_early[e][i] + 0.5);
        }
        const Stats ds = mean_stderr(diff);
        const Stats de = mean_stderr(diff_early);

        const double p_shape = auc(pool_pre[e], pool_post[e]);
        const double p_null = auc(pool_pre[e], pool_post_null[e]);

        std::printf("%-20s %3d/%-2d %7.3f+-%-6.3f %7.3f+-%-6.3f %7.3f+-%-6.3f %7.3f+-%-6.3f\n",
                    keys[e].c_str(), hits[e], n_seeds,
                    ss.mean, ss.se, ns.mean, ns.se, ds.mean, ds.se, de.mean, de.se);

        out << keys[e] << "," << hits[e] << "," << n_seeds << "," << ss.mean << "," << ss.se << ","
            << ns.mean << "," << ns.se << "," << ds.mean << "," << ds.se << ","
            << de.mean << "," << de.se << "," << p_shape << "," << p_null << ","
            << (p_shape - p_null + 0.5) << "," << hits_null[e] << "\n";
    }
    std::printf("\npooled (all seeds' blocks in one sample):\n");
    std::printf("%-20s %16s %16s %16s\n", "detector", "pooled shape", "pooled null", "pooled corr");
    for (std::size_t e = 0; e < E; ++e) {
        const double p_shape = auc(pool_pre[e], pool_post[e]);
        const double p_null = auc(pool_pre[e], pool_post_null[e]);
        std::printf("%-20s %16.3f %16.3f %16.3f\n", keys[e].c_str(), p_shape, p_null,
                    p_shape - p_null + 0.5);
    }
    std::printf("\nfalse alarms on the null run (sanity: should be low): ");
    for (std::size_t e = 0; e < E; ++e) std::printf("%s=%d ", keys[e].c_str(), hits_null[e]);
    std::printf("\nwrote results/auc_shape.csv and results/auc_blocks.csv\n");
    return 0;
}
