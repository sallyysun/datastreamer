// Asynchronous arrival vs optimal memory. See notes/studies/async_memory.md.
//
// A latent synchronous Gaussian-copula return process runs on a clock of T steps.
// Each leg is observed at its own Bernoulli(p) arrival times, pairs are formed by
// refresh-time pairing on actual timestamps (Chakrabarti & Sen 2022, algorithm A0),
// and each paired return is scaled by sqrt(its interval) so both margins stay N(0,1).
// Arms, all on the same latent path and the same refresh times:
//   sync   -- both legs observed at every refresh time (same pair rate, no overlap bias)
//   async  -- naive refresh-time pairing (overlap-attenuated)
//   corr   -- async output with the elliptical overlap correction theta = rho / mean(c)
// Every memory-knob estimator is swept over one effective-sample-size grid (in pairs)
// and scored against the true Spearman rho at every EVAL-th clock step.
//
// Usage: ./async_memory <constant|drift|away_pos|toward_pos|away_neg|toward_neg> [n_seeds] [base_seed]   (defaults: 20, 42)
// Writes results/async_memory_<scenario>.csv (long format) and prints M* per cell.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "memory_grid.hpp"
#include "trajectory_manifest.hpp"

namespace {

constexpr std::size_t T = 20000;
constexpr std::size_t BURN = 2000;
constexpr std::size_t EVAL = 10;
const std::vector<double> kGrid = {10, 20, 40, 80, 160, 320, 640, 1280, 2560};
using memgrid::kFamilies;
using memgrid::make_at_memory;
const std::vector<std::string> kArms = {"sync", "async", "corr"};

struct Liquidity { double px, py; };
const std::vector<Liquidity> kLiquidity = {{1.0, 1.0}, {0.5, 0.5}, {0.25, 0.25}, {0.1, 0.1}, {0.5, 0.1}};

double spearman_to_pearson(double rs) { return 2.0 * std::sin(std::numbers::pi * rs / 6.0); }
double pearson_to_spearman(double r) { return (6.0 / std::numbers::pi) * std::asin(std::clamp(r, -0.999, 0.999) / 2.0); }

// Directional drifts test the sign argument in the note: naive M* should shorten when
// rho drifts away from zero and lengthen when it drifts toward zero.
std::unique_ptr<Dependence> make_dependence(const std::string& scenario) {
    if (scenario == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (scenario == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, T);
    if (scenario == "away_pos") return std::make_unique<DriftDependence>(0.0, 0.8, T);
    if (scenario == "toward_pos") return std::make_unique<DriftDependence>(0.8, 0.0, T);
    if (scenario == "away_neg") return std::make_unique<DriftDependence>(0.0, -0.8, T);
    if (scenario == "toward_neg") return std::make_unique<DriftDependence>(-0.8, 0.0, T);
    return nullptr;
}

struct Latent {
    std::vector<double> z1, z2, rho_s;
};

Latent make_latent(const std::string& scenario, unsigned seed) {
    std::unique_ptr<Dependence> dep = make_dependence(scenario);
    auto copula = dgp::make_copula("gaussian");
    std::mt19937 rng(seed);
    Latent L;
    L.z1.resize(T); L.z2.resize(T); L.rho_s.resize(T);
    for (std::size_t t = 0; t < T; ++t) {
        const dgp::CopulaDraw d = copula->sample(std::clamp(dep->next(), -0.999, 0.999), rng);
        L.z1[t] = d.z1; L.z2[t] = d.z2; L.rho_s[t] = d.rho_s;
    }
    return L;
}

// cell index: [liquidity][arm][family][grid]
struct CellResult { double rmse = NAN, bias = NAN; };
struct SeedOut {
    std::vector<CellResult> cells;
    std::vector<double> pairs, steps_per_pair, mean_c;
};

std::size_t idx(std::size_t l, std::size_t a, std::size_t f, std::size_t g) {
    return ((l * kArms.size() + a) * kFamilies.size() + f) * kGrid.size() + g;
}

SeedOut run_seed(const std::string& scenario, unsigned seed) {
    const Latent L = make_latent(scenario, seed);
    SeedOut out;
    out.cells.resize(kLiquidity.size() * kArms.size() * kFamilies.size() * kGrid.size());

    for (std::size_t l = 0; l < kLiquidity.size(); ++l) {
        std::mt19937 arr(seed * 7919u + 1000u + static_cast<unsigned>(l));
        std::bernoulli_distribution ox(kLiquidity[l].px), oy(kLiquidity[l].py);

        // [arm 0 = sync, arm 1 = async][family][grid]; corr reads the async estimators.
        std::vector<std::unique_ptr<DependenceEstimator>> ests(2 * kFamilies.size() * kGrid.size());
        for (std::size_t a = 0; a < 2; ++a)
            for (std::size_t f = 0; f < kFamilies.size(); ++f)
                for (std::size_t g = 0; g < kGrid.size(); ++g)
                    ests[(a * kFamilies.size() + f) * kGrid.size() + g] = make_at_memory(kFamilies[f], kGrid[g]);

        std::vector<double> se(3 * ests.size() / 2, 0.0), be(se.size(), 0.0);
        std::size_t n_eval = 0;

        double px = 0.0, py = 0.0;                 // latent cumulative prices after step t
        long last_x_t = -1, last_y_t = -1;         // latest observation times
        double last_x_p = 0.0, last_y_p = 0.0;
        long pair_x_t = -1, pair_y_t = -1, pair_tau = -1;  // previous pair's times
        double pair_x_p = 0.0, pair_y_p = 0.0, pair_px = 0.0, pair_py = 0.0;
        bool new_x = false, new_y = false;
        std::size_t n_pairs = 0;
        double sum_c = 0.0;
        traj::Series ser;

        for (std::size_t t = 0; t < T; ++t) {
            px += L.z1[t];
            py += L.z2[t];
            if (ox(arr)) { last_x_t = static_cast<long>(t); last_x_p = px; new_x = true; }
            if (oy(arr)) { last_y_t = static_cast<long>(t); last_y_p = py; new_y = true; }

            if (new_x && new_y) {
                const long tau = static_cast<long>(t);
                if (pair_tau >= 0) {
                    const double Lx = static_cast<double>(last_x_t - pair_x_t);
                    const double Ly = static_cast<double>(last_y_t - pair_y_t);
                    const double overlap = std::max(0L, std::min(last_x_t, last_y_t) - std::max(pair_x_t, pair_y_t));
                    const double c = overlap / std::sqrt(Lx * Ly);
                    sum_c += c;
                    ++n_pairs;

                    const double Ls = static_cast<double>(tau - pair_tau);
                    const double sx = (px - pair_px) / std::sqrt(Ls), sy = (py - pair_py) / std::sqrt(Ls);
                    const double ax = (last_x_p - pair_x_p) / std::sqrt(Lx), ay = (last_y_p - pair_y_p) / std::sqrt(Ly);
                    const std::size_t half = ests.size() / 2;
                    for (std::size_t i = 0; i < half; ++i) {
                        if (ests[i]) ests[i]->update(sx, sy);
                        if (ests[half + i]) ests[half + i]->update(ax, ay);
                    }
                }
                pair_tau = tau;
                pair_px = px; pair_py = py;
                pair_x_t = last_x_t; pair_y_t = last_y_t;
                pair_x_p = last_x_p; pair_y_p = last_y_p;
                new_x = new_y = false;
            }

            if (t >= BURN && t % EVAL == 0) {
                const double truth = L.rho_s[t];
                const double c_hat = n_pairs ? sum_c / static_cast<double>(n_pairs) : 1.0;
                const std::size_t half = ests.size() / 2;
                for (std::size_t i = 0; i < half; ++i) {
                    if (!ests[i]) continue;
                    const double es = ests[i]->estimate();
                    const double ea = ests[half + i]->estimate();
                    ser.add(es);
                    ser.add(ea);
                    const double ec = pearson_to_spearman(spearman_to_pearson(ea) / std::max(c_hat, 1e-3));
                    const double d[3] = {es - truth, ea - truth, ec - truth};
                    for (std::size_t a = 0; a < 3; ++a) {
                        se[a * half + i] += d[a] * d[a];
                        be[a * half + i] += d[a];
                    }
                }
                ++n_eval;
            }
        }

        if (traj::on()) ser.commit(traj::tag("async_memory/%s/%u/l%zu/est", scenario.c_str(), seed, l));
        const std::size_t half = ests.size() / 2;
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t i = 0; i < half; ++i) {
                if (!ests[i]) continue;
                const std::size_t f = i / kGrid.size(), g = i % kGrid.size();
                out.cells[idx(l, a, f, g)] = {std::sqrt(se[a * half + i] / static_cast<double>(n_eval)),
                                              be[a * half + i] / static_cast<double>(n_eval)};
            }
        out.pairs.push_back(static_cast<double>(n_pairs));
        out.steps_per_pair.push_back(n_pairs ? static_cast<double>(T) / static_cast<double>(n_pairs) : NAN);
        out.mean_c.push_back(n_pairs ? sum_c / static_cast<double>(n_pairs) : NAN);
    }
    if (traj::on()) {
        std::vector<double> cells;
        for (const CellResult& c : out.cells) { cells.push_back(c.rmse); cells.push_back(c.bias); }
        const std::string base = traj::tag("async_memory/%s/%u/", scenario.c_str(), seed);
        traj::record(base + "cells", cells);
        traj::record(base + "pairs", out.pairs);
        traj::record(base + "steps_per_pair", out.steps_per_pair);
        traj::record(base + "mean_c", out.mean_c);
    }
    return out;
}

double mean_of(const std::vector<double>& v) {
    double s = 0.0; for (double x : v) s += x; return s / static_cast<double>(v.size());
}
double se_of(const std::vector<double>& v) {
    const double m = mean_of(v); double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1) / static_cast<double>(v.size()));
}

// Interpolated argmin on log2(M): parabola through the grid minimum and its neighbours.
// Returns the grid edge when the minimum sits on it (no interior optimum).
double argmin_memory(const std::vector<double>& M, const std::vector<double>& y, bool* at_edge) {
    std::size_t k = 0;
    for (std::size_t i = 1; i < y.size(); ++i) if (y[i] < y[k]) k = i;
    *at_edge = (k == 0 || k + 1 == y.size());
    if (*at_edge) return M[k];
    const double x0 = std::log2(M[k - 1]), x1 = std::log2(M[k]), x2 = std::log2(M[k + 1]);
    const double y0 = y[k - 1], y1 = y[k], y2 = y[k + 1];
    const double den = (x0 - x1) * (x0 - x2) * (x1 - x2);
    const double A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den;
    const double B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den;
    if (A <= 0.0) return M[k];
    return std::exp2(std::clamp(-B / (2.0 * A), x0, x2));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = (argc > 1) ? argv[1] : "drift";
    const int n_seeds = (argc > 2) ? std::atoi(argv[2]) : 20;
    const unsigned base = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42;
    if (!make_dependence(scenario)) {
        std::fprintf(stderr, "unknown scenario '%s' (use constant|drift|away_pos|toward_pos|away_neg|toward_neg)\n",
                     scenario.c_str());
        return 1;
    }

    std::vector<std::future<SeedOut>> jobs;
    for (int s = 0; s < n_seeds; ++s)
        jobs.push_back(std::async(std::launch::async, run_seed, scenario, base + static_cast<unsigned>(s)));
    std::vector<SeedOut> seeds;
    for (auto& j : jobs) seeds.push_back(j.get());

    const std::string path = "results/async_memory_" + scenario + ".csv";
    std::ofstream csv(path);
    if (!csv.is_open()) {
        std::fprintf(stderr, "cannot open '%s' (run: mkdir -p results)\n", path.c_str());
        return 1;
    }
    csv << "px,py,pairs,steps_per_pair,mean_c,arm,estimator,M_pairs,M_clock,rmse_mean,rmse_se,bias_mean,bias_se,n_seeds\n";

    std::printf("scenario=%s  n_seeds=%d  base_seed=%u  T=%zu  burn=%zu  (M in pairs; M_clock = M * steps/pair)\n",
                scenario.c_str(), n_seeds, base, T, BURN);

    std::mt19937 boot_rng(12345);
    for (std::size_t l = 0; l < kLiquidity.size(); ++l) {
        std::vector<double> pairs, spp, mc;
        for (const auto& s : seeds) { pairs.push_back(s.pairs[l]); spp.push_back(s.steps_per_pair[l]); mc.push_back(s.mean_c[l]); }
        const double SPP = mean_of(spp), C = mean_of(mc);
        std::printf("\n== px=%.2f py=%.2f  pairs=%.0f  steps/pair=%.2f  mean overlap c=%.3f\n",
                    kLiquidity[l].px, kLiquidity[l].py, mean_of(pairs), SPP, C);
        std::printf("%-6s %-16s %10s %10s %18s %9s %16s\n", "arm", "estimator", "M*_pairs", "M*_clock",
                    "M*_clock 68%CI", "RMSE*", "bias@M*");
        for (std::size_t a = 0; a < kArms.size(); ++a)
            for (std::size_t f = 0; f < kFamilies.size(); ++f) {
                std::vector<double> Ms, curve, curve_se, bias_at;
                std::vector<std::size_t> gs;
                for (std::size_t g = 0; g < kGrid.size(); ++g) {
                    if (std::isnan(seeds[0].cells[idx(l, a, f, g)].rmse)) continue;
                    std::vector<double> r, b;
                    for (const auto& s : seeds) { r.push_back(s.cells[idx(l, a, f, g)].rmse); b.push_back(s.cells[idx(l, a, f, g)].bias); }
                    Ms.push_back(kGrid[g]); curve.push_back(mean_of(r)); curve_se.push_back(se_of(r)); bias_at.push_back(mean_of(b));
                    gs.push_back(g);
                    csv << kLiquidity[l].px << "," << kLiquidity[l].py << "," << mean_of(pairs) << "," << SPP << "," << C << ","
                        << kArms[a] << "," << kFamilies[f] << "," << kGrid[g] << "," << kGrid[g] * SPP << ","
                        << mean_of(r) << "," << se_of(r) << "," << mean_of(b) << "," << se_of(b) << "," << n_seeds << "\n";
                }
                bool edge = false;
                const double mstar = argmin_memory(Ms, curve, &edge);

                std::vector<double> boot;
                for (int bi = 0; bi < 400; ++bi) {
                    std::vector<double> bc(Ms.size(), 0.0);
                    for (int s = 0; s < n_seeds; ++s) {
                        const auto& pick = seeds[boot_rng() % seeds.size()];
                        for (std::size_t k = 0; k < Ms.size(); ++k) bc[k] += pick.cells[idx(l, a, f, gs[k])].rmse;
                    }
                    bool e2 = false;
                    boot.push_back(argmin_memory(Ms, bc, &e2));
                }
                std::sort(boot.begin(), boot.end());
                const double lo = boot[boot.size() * 16 / 100] * SPP, hi = boot[boot.size() * 84 / 100] * SPP;

                std::size_t kmin = 0;
                for (std::size_t k = 1; k < curve.size(); ++k) if (curve[k] < curve[kmin]) kmin = k;
                char ci[64];
                std::snprintf(ci, sizeof ci, "[%.0f, %.0f]", lo, hi);
                std::printf("%-6s %-16s %9.0f%s %10.0f %18s %9.4f %+9.4f\n", kArms[a].c_str(), kFamilies[f].c_str(),
                            mstar, edge ? "|" : " ", mstar * SPP, ci, curve[kmin], bias_at[kmin]);
            }
    }
    std::printf("\n'|' = optimum on the grid edge (no interior M*).\nwrote %s\n", path.c_str());
    return 0;
}
