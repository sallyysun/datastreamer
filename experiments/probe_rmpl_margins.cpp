// Marginal diagnostics for the RMPL margin sensitivity study (chapter6_central_results.md, E2): on the streams of the
// probe_memory sweep (same seeds 42-81, matched paths), the EwMarginal pseudo-observations RMPL sees with decay
// (N_eff 500) and without (cumulative), against the true PIT Phi(x).
//   check:  EwMarginal(0) against the exact (R + 1)/(n + 2); EwMarginal(500) against the exact EW (B + w)/(W + 2w).
//   per scenario and margin: RMSE of U_hat - Phi(X) over t = 500-4999, and the normal scores z = probit(clip(U_hat))
//   that RMPL's score uses: mean z^2 (1 under the true PIT) and mean z1 z2 (vs the oracle's).
// Usage (from experiments/): SCENARIOS=matched ./probe_rmpl_margins -> results/rmpl_margins_2026-11-12.txt

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/ew_marginal.hpp"
#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "scenarios.hpp"

namespace {
constexpr std::size_t kN = 5000, kBurn = 500, kSeeds = 40;
constexpr unsigned kBase = 42;
constexpr double kDelta = 1e-3;   // RecursiveMPL's clip

double phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double clip(double u) { return std::clamp(u, kDelta, 1.0 - kDelta); }

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    return scenario_drift(kN);
}
}  // namespace

int main() {
    std::FILE* out = std::fopen("results/rmpl_margins_2026-11-12.txt", "w");
    if (!out) { std::fprintf(stderr, "run from experiments/\n"); return 1; }
    auto both = [&](const char* fmt, auto... a) { std::printf(fmt, a...); std::fprintf(out, fmt, a...); };

    // 1. Is EwMarginal(0) the cumulative empirical CDF, and EwMarginal(500) the EW one?
    {
        auto dep = make_dep("constant");
        auto cop = dgp::make_copula("gaussian");
        CopulaSource src(*dep, *cop, kBase);
        EwMarginal cum(0.0), ew(500.0);
        const double kap = 499.0 / 501.0;
        std::vector<double> xs;
        double dcum = 0, dew = 0;
        for (std::size_t t = 0; t < kN; ++t) {
            const double x = src.next().x;
            double r = 0, B = 0, W = 0, wt = 1;
            for (std::size_t i = xs.size(); i-- > 0;) {   // newest first: weight kappa^(t-i)
                wt *= kap;
                r += xs[i] <= x;
                B += wt * (xs[i] <= x);
                W += wt;
            }
            const double exact_cum = (r + 1.0) / (xs.size() + 2.0);
            const double exact_ew = (B + 1.0) / (W + 2.0);   // weights relative to x's own weight 1
            dcum = std::max(dcum, std::fabs(cum.transform(x) - exact_cum));
            dew = std::max(dew, std::fabs(ew.transform(x) - exact_ew));
            xs.push_back(x);
        }
        both("check (seed 42, 5000 obs): max |EwMarginal(0) - (R+1)/(n+2)| = %.4f; max |EwMarginal(500) - exact EW| = %.4f\n"
             "  (the summary's rank tolerance is eps = 0.005 of the weight, i.e. up to ~0.01 in u)\n\n", dcum, dew);
    }

    // 2. Marginal error and normal-score scale, per scenario and margin.
    both("%-9s %-14s %12s %12s %12s %12s\n", "scenario", "margin", "rmse(U-Phi)", "mean z^2", "mean z1z2", "z1z2/oracle");
    for (const std::string sc : {"constant", "drift"}) {
        double acc[3][4] = {};   // margin (ew500, ewcum, oracle) x {sq err, z^2, z1z2, count}
        for (std::size_t s = 0; s < kSeeds; ++s) {
            auto dep = make_dep(sc);
            auto cop = dgp::make_copula("gaussian");
            CopulaSource src(*dep, *cop, kBase + static_cast<unsigned>(s));
            EwMarginal ex[2] = {EwMarginal(500.0), EwMarginal(0.0)}, ey[2] = {EwMarginal(500.0), EwMarginal(0.0)};
            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                const double pu = phi(smp.x), pv = phi(smp.y);
                double u[3], v[3];
                for (int m = 0; m < 2; ++m) { u[m] = ex[m].transform(smp.x); v[m] = ey[m].transform(smp.y); }
                u[2] = pu; v[2] = pv;
                if (t < kBurn) continue;
                for (int m = 0; m < 3; ++m) {
                    const double z1 = RankPit::probit(clip(u[m])), z2 = RankPit::probit(clip(v[m]));
                    acc[m][0] += 0.5 * ((u[m] - pu) * (u[m] - pu) + (v[m] - pv) * (v[m] - pv));
                    acc[m][1] += 0.5 * (z1 * z1 + z2 * z2);
                    acc[m][2] += z1 * z2;
                    acc[m][3] += 1;
                }
            }
        }
        const char* nm[3] = {"EW 500", "cumulative EW", "oracle PIT"};
        for (int m = 0; m < 3; ++m)
            both("%-9s %-14s %12.5f %12.4f %12.4f %12.4f\n", sc.c_str(), nm[m], std::sqrt(acc[m][0] / acc[m][3]),
                 acc[m][1] / acc[m][3], acc[m][2] / acc[m][3], acc[m][2] / acc[2][2]);
    }
    std::fclose(out);
}
