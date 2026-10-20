// Explanatory ablation: does xiao_vff fail because its residual scale is wrong?
//
// Same estimator, rates, base memory, clamps, seeds and protocol as
// probe_adaptive.cpp's xiao vff rows. The ONLY change is the VFF's null
// variance: the fixed 1 that ships, against the simulator's true Var(p_t),
// p = 12(U-1/2)(V-1/2), at the current Gaussian-copula correlation. The oracle
// uses the DGP's truth, so it is an explanation, not a deployable method.
//
// Usage: ./probe_vff_norm   -> results/vff_norm{,_pairs,_traj}.csv

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "../src/estimators/dependence/xiao_spearman.hpp"
#include "../src/estimators/special_functions.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "trajectory_manifest.hpp"

namespace {

constexpr std::size_t kN = 5000;
constexpr std::size_t kBurn = 500;
constexpr std::size_t kSeeds = 40;
constexpr unsigned kBaseSeed = 42;
constexpr std::size_t kBins = 20;
constexpr double kVffBaseMem = 158.1;
constexpr double kT39 = 2.0227;  // t_{0.975, 39}

double alpha_from_ess(double e) { return 2.0 / (e + 1.0); }

// Var(p) for the Gaussian copula with linear correlation d, by Simpson's rule
// over (X, W) iid N(0,1), Y = dX + sqrt(1-d^2) W. Tabulated on a d grid.
class GaussianVarP {
public:
    GaussianVarP() {
        for (int i = 0; i <= kGrid; ++i) table_.push_back(compute(-kMax + 2.0 * kMax * i / kGrid));
    }
    double operator()(double d) const {
        const double f = std::clamp((d + kMax) / (2.0 * kMax), 0.0, 1.0) * kGrid;
        const int i = std::min(static_cast<int>(f), kGrid - 1);
        return table_[i] + (f - i) * (table_[i + 1] - table_[i]);
    }

private:
    static constexpr int kGrid = 400;
    static constexpr double kMax = 0.999;

    static double compute(double d) {
        constexpr int n = 401;
        constexpr double lo = -8.0, hi = 8.0, h = (hi - lo) / (n - 1);
        const double s = std::sqrt(1.0 - d * d);
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            const double x = lo + i * h;
            const double wi = (i == 0 || i == n - 1) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            const double gx = std::exp(-0.5 * x * x);
            const double ux = special::normal_cdf(x) - 0.5;
            double inner = 0.0;
            for (int j = 0; j < n; ++j) {
                const double w = lo + j * h;
                const double wj = (j == 0 || j == n - 1) ? 1.0 : (j % 2 ? 4.0 : 2.0);
                const double uy = special::normal_cdf(d * x + s * w) - 0.5;
                inner += wj * std::exp(-0.5 * w * w) * uy * uy;
            }
            acc += wi * gx * ux * ux * inner;
        }
        const double e = acc * (h / 3.0) * (h / 3.0) / (2.0 * std::numbers::pi);
        const double rho_s = 6.0 / std::numbers::pi * std::asin(d / 2.0);
        return 144.0 * e - rho_s * rho_s;
    }

    std::vector<double> table_;
};

std::unique_ptr<Dependence> make_dependence(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kN / 2);
    if (s == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

struct Cfg { double rate; bool oracle; const char* name; };

// Per-tick channels recorded for the trajectory export.
enum Ch { kEst, kMem, kEss, kMs, kDen, kRatio, kC, kNumCh };

void run_seed(const std::string& scenario, const std::vector<Cfg>& cfgs, std::size_t s,
              const GaussianVarP& varp,
              std::vector<std::vector<std::vector<std::vector<double>>>>& rec,
              std::vector<double>& truth, bool record_truth) {
    auto dep = make_dependence(scenario);
    auto cop = dgp::make_copula("gaussian");
    CopulaSource src(*dep, *cop, kBaseSeed + static_cast<unsigned>(s));

    std::vector<std::unique_ptr<XiaoSpearmanEstimator>> ests;
    for (const Cfg& c : cfgs)
        ests.push_back(std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = kBins, .window = 0,
                                          .lambda = 1.0 - alpha_from_ess(kVffBaseMem),
                                          .calibration = 100, .vff_rate = c.rate}, "m"));
    std::vector<double> s1(cfgs.size(), 0.0), s2(cfgs.size(), 0.0);
    for (std::size_t c = 0; c < cfgs.size(); ++c)
        for (int ch = 0; ch < kNumCh; ++ch) rec[c][ch][s].resize(kN);

    for (std::size_t t = 0; t < kN; ++t) {
        const CopulaSource::Sample smp = src.next();
        if (record_truth) truth[t] = smp.rho_s;
        for (std::size_t c = 0; c < cfgs.size(); ++c) {
            XiaoSpearmanEstimator& e = *ests[c];
            if (cfgs[c].oracle) e.set_vff_null_var(varp(smp.rho));
            e.update(smp.x, smp.y);
            const double mem = e.effective_memory_ticks();
            const double a = alpha_from_ess(mem);
            // Kish ESS of the weights actually applied: S1 = sum w, S2 = sum w^2.
            s1[c] = (1.0 - a) * s1[c] + a;
            s2[c] = (1.0 - a) * (1.0 - a) * s2[c] + a * a;
            auto& r = rec[c];
            r[kEst][s][t] = e.estimate();
            r[kMem][s][t] = mem;
            r[kEss][s][t] = s2[c] > 0.0 ? s1[c] * s1[c] / s2[c] : 0.0;
            r[kMs][s][t] = e.vff_last_ms();
            r[kDen][s][t] = e.vff_last_denom();
            r[kRatio][s][t] = e.vff_last_denom() > 0.0 ? e.vff_last_ms() / e.vff_last_denom() : 0.0;
            r[kC][s][t] = e.vff_c();
        }
    }
}

double seed_rmse(const std::vector<double>& est, const std::vector<double>& truth) {
    double acc = 0.0;
    for (std::size_t t = kBurn; t < kN; ++t) acc += (est[t] - truth[t]) * (est[t] - truth[t]);
    return std::sqrt(acc / static_cast<double>(kN - kBurn));
}

}  // namespace

int main() {
    const GaussianVarP varp;
    std::printf("oracle Var(p): d=0 %.4f  d=0.2 %.4f  d=0.6 %.4f  d=0.8 %.4f\n",
                varp(0.0), varp(0.2), varp(0.6), varp(0.8));

    if (traj::on()) {
        std::vector<double> vp;
        for (int i = -999; i <= 999; ++i) vp.push_back(varp(i / 1000.0));
        traj::record("probe_vff_norm/varp_grid", vp);
    }

    const std::vector<Cfg> cfgs{{0.05, false, "vff_slow_fixed1"}, {0.05, true, "vff_slow_oracle"},
                                {0.5, false, "vff_fast_fixed1"},  {0.5, true, "vff_fast_oracle"}};
    const std::vector<std::string> scenarios{"constant", "drift", "jump", "oscillate"};

    std::ofstream out("results/vff_norm.csv");
    std::ofstream pout("results/vff_norm_pairs.csv");
    std::ofstream tout("results/vff_norm_traj.csv");
    if (!out.is_open() || !pout.is_open() || !tout.is_open()) {
        std::fprintf(stderr, "cannot write results/ (run: mkdir -p results)\n");
        return 1;
    }
    out << "scenario,config,rmse_pooled,rmse_mean_seed,se\n";
    pout << "scenario,rate,rmse_fixed1,rmse_oracle,diff_mean,diff_se,ci_lo,ci_hi\n";
    tout << "scenario,config,t,rho_hat,mem_nominal,ess_kish,ms_raw,denom,ratio,C,truth\n";

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (const std::string& scenario : scenarios) {
        std::vector<std::vector<std::vector<std::vector<double>>>> rec(
            cfgs.size(), std::vector<std::vector<std::vector<double>>>(
                             kNumCh, std::vector<std::vector<double>>(kSeeds)));
        std::vector<double> truth(kN);
        run_seed(scenario, cfgs, 0, varp, rec, truth, true);
        std::vector<std::thread> pool;
        std::vector<std::vector<double>> ignored(hw, std::vector<double>(kN));
        for (unsigned w = 0; w < hw; ++w)
            pool.emplace_back([&, w] {
                for (std::size_t s = 1 + w; s < kSeeds; s += hw)
                    run_seed(scenario, cfgs, s, varp, rec, ignored[w], false);
            });
        for (auto& th : pool) th.join();
        if (traj::on()) {
            static const char* kChName[kNumCh] = {"est", "mem", "ess", "ms", "den", "ratio", "C"};
            traj::record("probe_vff_norm/" + scenario + "/truth", truth);
            for (std::size_t c = 0; c < cfgs.size(); ++c)
                for (std::size_t s = 0; s < kSeeds; ++s)
                    for (int ch = 0; ch < kNumCh; ++ch)
                        traj::record(traj::tag("probe_vff_norm/%s/%s/%zu/%s", scenario.c_str(), cfgs[c].name, s,
                                               kChName[ch]),
                                     rec[c][ch][s]);
        }

        std::vector<std::vector<double>> per_seed(cfgs.size(), std::vector<double>(kSeeds));
        for (std::size_t c = 0; c < cfgs.size(); ++c) {
            double pooled = 0.0, m = 0.0;
            for (std::size_t s = 0; s < kSeeds; ++s) {
                per_seed[c][s] = seed_rmse(rec[c][kEst][s], truth);
                m += per_seed[c][s];
                for (std::size_t t = kBurn; t < kN; ++t) {
                    const double d = rec[c][kEst][s][t] - truth[t];
                    pooled += d * d;
                }
            }
            m /= kSeeds;
            double v = 0.0;
            for (double x : per_seed[c]) v += (x - m) * (x - m);
            out << scenario << "," << cfgs[c].name << ","
                << std::sqrt(pooled / (kSeeds * static_cast<double>(kN - kBurn))) << "," << m << ","
                << std::sqrt(v / (kSeeds - 1.0) / kSeeds) << "\n";
        }
        if (traj::on())
            for (std::size_t c = 0; c < cfgs.size(); ++c)
                traj::record(traj::tag("probe_vff_norm/%s/%s/seed_rmse", scenario.c_str(), cfgs[c].name), per_seed[c]);
        for (std::size_t c = 0; c + 1 < cfgs.size(); c += 2) {
            double m = 0.0;
            for (std::size_t s = 0; s < kSeeds; ++s) m += per_seed[c + 1][s] - per_seed[c][s];
            m /= kSeeds;
            double v = 0.0;
            for (std::size_t s = 0; s < kSeeds; ++s) {
                const double e = per_seed[c + 1][s] - per_seed[c][s] - m;
                v += e * e;
            }
            const double se = std::sqrt(v / (kSeeds - 1.0) / kSeeds);
            double a = 0.0, b = 0.0;
            for (std::size_t s = 0; s < kSeeds; ++s) { a += per_seed[c][s]; b += per_seed[c + 1][s]; }
            pout << scenario << "," << cfgs[c].rate << "," << a / kSeeds << "," << b / kSeeds << ","
                 << m << "," << se << "," << m - kT39 * se << "," << m + kT39 * se << "\n";
        }
        for (std::size_t c = 0; c < cfgs.size(); ++c)
            for (std::size_t t = 0; t < kN; t += 10) {
                double acc[kNumCh] = {};
                for (std::size_t s = 0; s < kSeeds; ++s)
                    for (int ch = 0; ch < kNumCh; ++ch) acc[ch] += rec[c][ch][s][t];
                tout << scenario << "," << cfgs[c].name << "," << t;
                for (int ch = 0; ch < kNumCh; ++ch) tout << "," << acc[ch] / kSeeds;
                tout << "," << truth[t] << "\n";
            }
        std::printf("scenario=%s done\n", scenario.c_str());
    }
    std::printf("wrote results/vff_norm{,_pairs,_traj}.csv\n");
    return 0;
}
