// Working-family scoring rules and family identification at matched Spearman dependence.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "strategy_results.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

using thesis::H2Table;
using thesis::Win;
using thesis::checkpoint;
using thesis::kBurn;
using thesis::kChange;
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
            const StudySeries s = load_study_series(cop, sc, seed, kN);
            const auto generating_parameter = generating_parameters(cop, s);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) lc[t] = copula::log_density(h2.F, s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t]);
            // inversion estimators: arm forecasts once
            std::vector<std::vector<std::vector<double>>> FC(3, std::vector<std::vector<double>>(kK, std::vector<double>(kN)));
            std::vector<std::vector<char>> RD(3, std::vector<char>(kN));
            for (int ei = 0; ei < 3; ++ei) {
                FixedMemoryArms arms(EN[ei]);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) FC[ei][k][t] = arms.estimate(k);
                    RD[ei][t] = arms.ready();
                    arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
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
                            ld[k] = copula::log_density(WF[wi], s.true_pit_x[t], s.true_pit_y[t], evaluation_parameters[k][t]);
                            if (!std::isfinite(ld[k])) { ++enf; ld[k] = -INFINITY; }
                        }
                        const bool cp = !noh2 && checkpoint(sc, t);
                        for (auto& q : in) {
                            double r = 0, top = -INFINITY;
                            for (std::size_t k = 0; k < kK; ++k) { r += q.w[k] * rk[k]; if (q.w[k] > 0) top = std::max(top, std::log(q.w[k]) + ld[k]); }
                            double m = 0;
                            for (std::size_t k = 0; k < kK; ++k) if (q.w[k] > 0) m += std::exp(std::log(q.w[k]) + ld[k] - top);
                            r = std::clamp(r, -1.0, 1.0);
                            q.se[t] = (r - s.true_spearman[t]) * (r - s.true_spearman[t]);
                            q.ex[t] = lc[t] - (top + std::log(m));
                            if (cp) q.hh[t] = hellinger.single_working_family(wi, generating_parameter[t], rk.data(), q.w.data(), kK);
                        }
                        if (t >= 30 && ready[t]) {
                            const double p = 12.0 * (s.estimated_pit_x[t] - 0.5) * (s.estimated_pit_y[t] - 0.5);
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - rk[k]) * (p - rk[k]);
                                llog[k] = -copula::log_density(WF[wi], s.estimated_pit_x[t], s.estimated_pit_y[t], update_parameters[k][t]);
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

// Compare family-aware and Gaussian-only MFF at matched dependence. Families switch at tick 2500.
int run_famswitch(const Run& run) {
    const int seeds = run.seeds;
    H2Table& h2 = run.h2;
    const char* GN[7] = {"gaussian", "t", "clayton", "gumbel", "gauss2t", "gauss2clayton", "clayton2gumbel"};
    const int G0[7] = {0, 1, 2, 3, 0, 0, 2}, G1[7] = {0, 1, 2, 3, 1, 2, 3};
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const FamilyInverse inv[4] = {FamilyInverse(WF[0]), FamilyInverse(WF[1]), FamilyInverse(WF[2]), FamilyInverse(WF[3])};
    auto theta_of = [&](int wi, double rho) { return inv[wi](rho); };
    const std::string fx = std::getenv("FS_TAG") ? std::string("_") + std::getenv("FS_TAG") : std::string();
    std::FILE* fo = std::fopen(("results/famsw_seeds" + fx + ".csv").c_str(), "w");
    std::FILE* fi = std::fopen(("results/famsw_ident" + fx + ".csv").c_str(), "w");
    std::FILE* fp = std::fopen(("results/famsw_path" + fx + ".csv").c_str(), "w");
    if (!fo || !fi || !fp) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    // FS_H2 enables squared Hellinger at the Chapter 6 checkpoints.
    const bool fh2 = std::getenv("FS_H2") != nullptr;
    std::fprintf(fo, fh2 ? "gen,seed,method,window,rmse,excess_x1000,h2_x1000\n" : "gen,seed,method,window,rmse,excess_x1000\n");
    FamilyHellinger hellinger(h2, inv, fh2);
    if (fh2) {   // tabulated components vs exact dmetric integration, one mixture per non-Gaussian family
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi) {
            const int wis[3] = {wi, wi, 0}; const double rhs[3] = {0.31, 0.6123, 0.55}, wss[3] = {0.3, 0.5, 0.2};
            const double par = copula::param_from_spearman(WF[wi], thesis::kConstantRhoS);
            const dmetric::Table tt(h2.g, [&](double a, double b) { const double l = copula::log_density(WF[wi], a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 3; ++k) { const double th = wis[k] == 0 ? copula::param_from_spearman(copula::Family::Gaussian, rhs[k]) : copula::param_from_spearman(WF[wis[k]], rhs[k]); const double l = copula::log_density(WF[wis[k]], a, b, th); m += wss[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
            const double ex = dmetric::hellinger2(h2.g, tt, mt);
            const double tb = hellinger.mixed_families(WF[wi], par, wis, rhs, wss, 3);
            worst = std::max(worst, std::fabs(tb - ex) / ex);
        }
        if (study_verbose())
            std::printf("Family-switch mixture H2 max relative error %.2e\n", worst);
    }
    std::fprintf(fi, "gen,seed,kind,target,t50,t90,share_ge50_after,incorrect_majority,first_incorrect,pre_weight_target,n_guard\n");
    std::fprintf(fp, "gen,t,family,mean,sd\n");
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : thesis::kFirstSeed;
    // FS_EST = rmpl (default) | xiao | hermite: the five arms of that estimator (counting matrix / Hermite read the raw
    // data with their own margins, as in Sections 6.3-6.4; RMPL reads probit of the KLL pseudo-observations).
    const std::string fest = std::getenv("FS_EST") ? std::getenv("FS_EST") : "rmpl";
    std::vector<int> gens;
    if (const char* gs = std::getenv("FS_GENS")) {   // comma-separated subset of GN
        std::string s(gs); std::size_t p = 0;
        while (p <= s.size()) {
            std::size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            const std::string tok = s.substr(p, q - p);
            for (int g = 0; g < 7; ++g) if (tok == GN[g]) gens.push_back(g);
            p = q + 1;
        }
    } else for (int g = 0; g < 7; ++g) gens.push_back(g);
    for (int g : gens) {
        const bool sw = G0[g] != G1[g];
        std::vector<double> ps(4 * kN, 0.0), pq(4 * kN, 0.0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const StudySeries s = load_study_series(GN[g], "constant", seed, kN);
            FixedMemoryArms arms(fest);
            // instance 0: family (20 models, index wi * 5 + k); instance 1: Gaussian-only (5 models)
            std::vector<double> wf(20, 1.0 / 20), wg(kK, 1.0 / kK);
            std::vector<double> sef(kN), seg(kN), exf(kN), exg(kN), hhf(kN, NAN), hhg(kN, NAN);
            std::vector<std::array<double, 4>> fw(kN);
            long nguard = 0;
            double th[4][kK], te[4][kK], ld[4][kK], f[kK];
            std::vector<double> family_losses(wf.size()), gaussian_losses(wg.size()), log_weights(wf.size());
            for (std::size_t t = 0; t < kN; ++t) {
                arms.forecasts(f);
                const bool ready = t >= 30 && arms.ready();
                const int gt = t < kChange ? G0[g] : G1[g];
                const double part = copula::param_from_spearman(WF[gt], thesis::kConstantRhoS);
                const double lc = copula::log_density(WF[gt], s.true_pit_x[t], s.true_pit_y[t], part);
                for (int wi = 0; wi < 4; ++wi)
                    for (std::size_t k = 0; k < kK; ++k) {
                        th[wi][k] = theta_of(wi, f[k]);
                        te[wi][k] = wi == 0 ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(f[k], -0.999, 0.999)) : th[wi][k];
                        ld[wi][k] = copula::log_density(WF[wi], s.true_pit_x[t], s.true_pit_y[t], te[wi][k]);
                    }
                // forecasts held before tick t
                auto mixlog = [&](const double* w, int nw, bool fam) {
                    double top = -INFINITY;
                    for (int i = 0; i < nw; ++i) if (w[i] > 0) top = std::max(top, std::log(w[i]) + (fam ? ld[i / kK][i % kK] : ld[0][i]));
                    double m = 0;
                    for (int i = 0; i < nw; ++i) if (w[i] > 0) m += std::exp(std::log(w[i]) + (fam ? ld[i / kK][i % kK] : ld[0][i]) - top);
                    return top + std::log(m);
                };
                double rf = 0, rg = 0;
                for (int i = 0; i < 20; ++i) rf += wf[i] * f[i % kK];
                for (std::size_t k = 0; k < kK; ++k) rg += wg[k] * f[k];
                rf = std::clamp(rf, -1.0, 1.0); rg = std::clamp(rg, -1.0, 1.0);
                sef[t] = (rf - s.true_spearman[t]) * (rf - s.true_spearman[t]); seg[t] = (rg - s.true_spearman[t]) * (rg - s.true_spearman[t]);
                exf[t] = lc - mixlog(wf.data(), 20, true); exg[t] = lc - mixlog(wg.data(), kK, false);
                if (fh2 && checkpoint(sw ? "jump" : "constant", t)) {
                    int wis[20]; double rhs[20];
                    for (int i = 0; i < 20; ++i) { wis[i] = i / int(kK); rhs[i] = f[i % kK]; }
                    hhf[t] = hellinger.mixed_families(WF[gt], part, wis, rhs, wf.data(), wf.size());
                    hhg[t] = hellinger.mixed_families(WF[gt], part, wis, rhs, wg.data(), wg.size());
                }
                for (int wi = 0; wi < 4; ++wi) { double a = 0; for (std::size_t k = 0; k < kK; ++k) a += wf[wi * kK + k]; fw[t][wi] = a; }
                if (ready) {
                    for (int wi = 0; wi < 4; ++wi) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            family_losses[wi * kK + k] = -copula::log_density(WF[wi], s.estimated_pit_x[t], s.estimated_pit_y[t], th[wi][k]);
                        }
                    }
                    std::copy_n(family_losses.begin(), kK, gaussian_losses.begin());
                    nguard += thesis::guard_mff_losses(family_losses);
                    nguard += thesis::guard_mff_losses(gaussian_losses);
                    thesis::apply_fixed_share(wf, family_losses, log_weights, 1.0, 1e-4);
                    thesis::apply_fixed_share(wg, gaussian_losses, log_weights, 1.0, 1e-4);
                }
                arms.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t]);
            }
            // windows
            std::vector<Win> wins{{"whole", kBurn, kN}};
            if (sw) { wins.push_back({"pre", kBurn, kChange}); wins.push_back({"rec_0_100", kChange, kChange + 100});
                      wins.push_back({"late", kChange + 100, kN}); wins.push_back({"post", kChange, kN}); }
            for (const auto& w : wins)
                for (int m = 0; m < 2; ++m) {
                    const auto [rmse, excess, h2_score] = summarize_window(w, m ? seg : sef, m ? exg : exf, m ? hhg : hhf);
                    if (fh2) std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, rmse, excess, h2_score);
                    else std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, rmse, excess);
                }
            // identification: constant = true family from t = 0; switch = new family from 2500 (times relative)
            const int tgt = G1[g];
            const std::size_t t0 = sw ? kChange : 0;
            long t50 = -1, t90 = -1, firstinc = -1; long after = 0, ge = 0; bool inc = false;
            for (std::size_t t = t0; t < kN; ++t) {
                const double wt = fw[t][tgt];
                if (t50 < 0 && wt >= 0.5) t50 = long(t - t0);
                if (t90 < 0 && wt >= 0.9) t90 = long(t - t0);
                if (t50 >= 0 && long(t - t0) >= t50) {
                    ++after; if (wt >= 0.5) ++ge;
                    for (int wi = 0; wi < 4; ++wi) if (wi != tgt && fw[t][wi] >= 0.5 && !inc) { inc = true; firstinc = long(t - t0); }
                }
            }
            std::fprintf(fi, "%s,%u,%s,%s,%ld,%ld,%.6f,%d,%ld,%.6f,%ld\n", GN[g], seed, sw ? "switch" : "initial", WN[tgt], t50, t90,
                         after ? double(ge) / after : NAN, int(inc), firstinc, sw ? fw[kChange - 1][tgt] : fw[0][tgt], nguard);
            for (std::size_t t = 0; t < kN; ++t) for (int wi = 0; wi < 4; ++wi) { ps[wi * kN + t] += fw[t][wi]; pq[wi * kN + t] += fw[t][wi] * fw[t][wi]; }
        }
        for (std::size_t t = 0; t < kN; t += 10)
            for (int wi = 0; wi < 4; ++wi) {
                const double mu = ps[wi * kN + t] / seeds, var = std::max(0.0, pq[wi * kN + t] / seeds - mu * mu) * seeds / (seeds - 1.0);
                std::fprintf(fp, "%s,%zu,%s,%.6f,%.6f\n", GN[g], t, WN[wi], mu, std::sqrt(var));
            }
    }
    std::fclose(fo); std::fclose(fi); std::fclose(fp);
    return 0;
}

}  // namespace strategy_experiments
