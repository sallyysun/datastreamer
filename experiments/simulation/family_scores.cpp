// Squared and log scoring rules across working copula families.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "family_hellinger.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

using thesis::H2Table;
using thesis::checkpoint;
using thesis::kBurn;
using thesis::kK;
using thesis::kN;
using thesis::windows;

// Compare squared and log loss across working copulas, using the same KLL inputs and arms.
int run_scoresfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : thesis::kFirstSeed;
    const bool noh2 = std::getenv("NOH2") != nullptr;
    std::vector<double> etas{1.0};
    if (const char* es = std::getenv("ETAS")) {
        etas.clear(); std::string s(es); std::size_t p = 0;
        while (p < s.size()) { std::size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); etas.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    if (!tag) { std::fprintf(stderr, "SCORESFAM needs TAG\n"); return 1; }
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const FamilyInverse inv[4] = {FamilyInverse(WF[0]), FamilyInverse(WF[1]), FamilyInverse(WF[2]), FamilyInverse(WF[3])};
    {   // Compare the fine inverse grid with the shared family conversion.
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (double r = -0.2; r < 0.995; r += 0.0137) {
                const double a = inv[wi](r), b = copula::param_from_spearman(WF[wi], r);
                worst = std::max(worst, std::fabs(a - b) / std::max(1e-3, std::fabs(b)));
            }
        if (study_verbose())
            std::printf("Inverse table max relative error %.2e\n", worst);
    }
    FamilyHellinger hellinger(h2, inv, !noh2);
    if (!noh2) {   // tabulated mixture H^2 vs an exact dmetric table of the same mixture
        const double par = copula::param_from_spearman(h2.F, 0.55);
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (const auto& rs : std::vector<std::array<double, 5>>{{0.21, 0.4567, 0.6, 0.78, 0.91}, {0.05, 0.33, 0.52, 0.5501, 0.97}, {-0.2, 0.1, 0.3, 0.7, 0.85}}) {
                const double ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15};
                double th[5];
                for (int k = 0; k < 5; ++k) th[k] = inv[wi](rs[k]);
                const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
                const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) { const double l = copula::log_density(WF[wi], a, b, th[k]); m += ws[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
                const double ex = dmetric::hellinger2(h2.g, truth, mt);
                const double tb = hellinger.single_working_family(wi, par, rs.data(), ws, 5);
                worst = std::max(worst, std::fabs(tb - ex) / ex);
            }
        if (study_verbose())
            std::printf("Mixture H2 max relative error %.2e\n", worst);
    }
    std::FILE* fo = std::fopen(("results/scoresfam_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fd = std::fopen(("results/scoresfam_diag" + sfx + ".csv").c_str(), "w");
    if (!fo || !fd) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fo, "gen,scenario,estimator,working,score,eta,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fd, "gen,scenario,estimator,working,seed,arm_clamp_ticks,score_nonfinite,eval_nonfinite\n");
    const char* EN[3] = {"rmpl", "xiao", "hermite"};
    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) lc[t] = copula::log_density(h2.F, s.pu[t], s.pv[t], s.par[t]);
            // inversion estimators: arm forecasts once
            std::vector<std::vector<std::vector<double>>> FC(3, std::vector<std::vector<double>>(kK, std::vector<double>(kN)));
            std::vector<std::vector<char>> RD(3, std::vector<char>(kN));
            for (int ei = 0; ei < 3; ++ei) {
                FixedMemoryArms arms(EN[ei]);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) FC[ei][k][t] = arms.estimate(k);
                    RD[ei][t] = arms.ready();
                    arms.update(s, t);
                }
            }
            for (int wi = 0; wi < 4; ++wi) {
                for (int ei = 0; ei < 3; ++ei) {
                    const auto& forecasts = FC[ei];
                    const auto& ready = RD[ei];
                    std::vector<std::vector<double>> update_parameters(kK, std::vector<double>(kN));
                    std::vector<std::vector<double>> evaluation_parameters(kK, std::vector<double>(kN));
                    long aclamp = 0;
                    for (std::size_t k = 0; k < kK; ++k)
                        for (std::size_t t = 0; t < kN; ++t) {
                            const double rho = forecasts[k][t];
                            const double parameter = inv[wi](rho);
                            update_parameters[k][t] = parameter;
                            const bool clamped = wi == 0 ? std::fabs(parameter) >= copula::Gaussian::kMaxParam
                                                         : inv[wi].clamped(rho);
                            if (clamped && t >= kBurn) ++aclamp;
                            evaluation_parameters[k][t] = wi == 0
                                ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rho, -0.999, 0.999)) : parameter;
                        }
                    struct In { int score; double eta; std::vector<double> w; std::vector<double> se, ex, hh; };
                    std::vector<In> in;
                    for (double h : etas) for (int sco = 0; sco < 2; ++sco)
                        in.push_back({sco, h, std::vector<double>(kK, 1.0 / kK), std::vector<double>(kN), std::vector<double>(kN), std::vector<double>(kN, NAN)});
                    long snf = 0, enf = 0;
                    std::vector<double> lsq(kK), llog(kK), ld(kK), lw(kK), rk(kK);
                    for (std::size_t t = 0; t < kN; ++t) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            rk[k] = forecasts[k][t];
                            ld[k] = copula::log_density(WF[wi], s.pu[t], s.pv[t], evaluation_parameters[k][t]);
                            if (!std::isfinite(ld[k])) { ++enf; ld[k] = -INFINITY; }
                        }
                        const bool cp = !noh2 && checkpoint(sc, t);
                        for (auto& q : in) {
                            double r = 0, top = -INFINITY;
                            for (std::size_t k = 0; k < kK; ++k) { r += q.w[k] * rk[k]; if (q.w[k] > 0) top = std::max(top, std::log(q.w[k]) + ld[k]); }
                            double m = 0;
                            for (std::size_t k = 0; k < kK; ++k) if (q.w[k] > 0) m += std::exp(std::log(q.w[k]) + ld[k] - top);
                            r = std::clamp(r, -1.0, 1.0);
                            q.se[t] = (r - s.rho[t]) * (r - s.rho[t]);
                            q.ex[t] = lc[t] - (top + std::log(m));
                            if (cp) q.hh[t] = hellinger.single_working_family(wi, s.par[t], rk.data(), q.w.data(), kK);
                        }
                        if (t >= 30 && ready[t]) {
                            const double p = 12.0 * (s.u[t] - 0.5) * (s.v[t] - 0.5);
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - rk[k]) * (p - rk[k]);
                                llog[k] = -copula::log_density(WF[wi], s.u[t], s.v[t], update_parameters[k][t]);
                            }
                            snf += thesis::guard_mff_losses(llog);
                            for (auto& q : in) {
                                thesis::apply_fixed_share(q.w, q.score ? llog : lsq, lw, q.eta, 1e-4);
                            }
                        }
                    }
                    for (const auto& q : in)
                        for (const auto& w : wins) {
                            const auto [rmse, excess, h2_score] = summarize_window(w, q.se, q.ex, q.hh);
                            std::fprintf(fo, "%s,%s,%s,%s,%s,%g,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), EN[ei], WN[wi], q.score ? "log" : "squared",
                                         q.eta, seed, w.name, rmse, excess, h2_score);
                        }
                    std::fprintf(fd, "%s,%s,%s,%s,%u,%ld,%ld,%ld\n", cop.c_str(), sc.c_str(), EN[ei], WN[wi], seed, aclamp, snf, enf);
                }
            }
        }
        std::fflush(stdout);
    }
    std::fclose(fo); std::fclose(fd);
    return 0;
}

}  // namespace strategy_experiments
