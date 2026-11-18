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
#include "working_families.hpp"
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
        etas.clear();
        for (const auto& eta : parse_list(es)) etas.push_back(std::atof(eta.c_str()));
    }
    if (!tag) { std::fprintf(stderr, "SCORESFAM needs TAG\n"); return 1; }
    const WorkingFamilies families;
    FamilyHellinger hellinger(h2, families.inverse, !noh2);
    std::FILE* fo = open_result("scoresfam_seeds" + sfx);
    std::FILE* fd = open_result("scoresfam_diag" + sfx);
    if (!fo || !fd) return missing_results_folder();
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
                            const double parameter = families.inverse[wi](rho);
                            update_parameters[k][t] = parameter;
                            const bool clamped = wi == 0 ? std::fabs(parameter) >= copula::Gaussian::kMaxParam
                                                         : families.inverse[wi].clamped(rho);
                            if (clamped && t >= kBurn) ++aclamp;
                            evaluation_parameters[k][t] = families.evaluation_parameter(wi, rho);
                        }
                    struct In { int score; thesis::Mff mixture; std::vector<double> se, ex, hh; };
                    std::vector<In> in;
                    for (double h : etas) for (int sco = 0; sco < 2; ++sco)
                        in.push_back({sco, thesis::Mff(copula::Family::Gaussian, kK, h), std::vector<double>(kN), std::vector<double>(kN), std::vector<double>(kN, NAN)});
                    long snf = 0, enf = 0;
                    std::vector<double> lsq(kK), llog(kK), ld(kK), rk(kK);
                    for (std::size_t t = 0; t < kN; ++t) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            rk[k] = forecasts[k][t];
                            ld[k] = copula::log_density(WorkingFamilies::family[wi], s.true_pit_x[t], s.true_pit_y[t], evaluation_parameters[k][t]);
                            if (!std::isfinite(ld[k])) { ++enf; ld[k] = -INFINITY; }
                        }
                        const bool cp = !noh2 && checkpoint(sc, t);
                        for (auto& q : in) {
                            const double r = q.mixture.forecast(rk.data());
                            q.se[t] = (r - s.true_spearman[t]) * (r - s.true_spearman[t]);
                            q.ex[t] = lc[t] - q.mixture.mixture_log_density(ld.data());
                            if (cp) q.hh[t] = hellinger.single_working_family(wi, generating_parameter[t], rk.data(), q.mixture.w.data(), kK);
                        }
                        if (t >= thesis::kMffWarmup && ready[t]) {
                            const double p = 12.0 * (s.estimated_pit_x[t] - 0.5) * (s.estimated_pit_y[t] - 0.5);
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - rk[k]) * (p - rk[k]);
                                llog[k] = -copula::log_density(WorkingFamilies::family[wi], s.estimated_pit_x[t], s.estimated_pit_y[t], update_parameters[k][t]);
                            }
                            snf += thesis::guard_mff_losses(llog);
                            for (auto& q : in) q.mixture.update(q.score ? llog : lsq);
                        }
                    }
                    for (const auto& q : in)
                        for (const auto& w : wins) {
                            const auto [rmse, excess, h2_score] = summarize_window(w, q.se, q.ex, q.hh);
                            std::fprintf(fo, "%s,%s,%s,%s,%s,%g,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), EN[ei], WorkingFamilies::name[wi], q.score ? "log" : "squared",
                                         q.mixture.eta, seed, w.name, rmse, excess, h2_score);
                        }
                    std::fprintf(fd, "%s,%s,%s,%s,%u,%ld,%ld,%ld\n", cop.c_str(), sc.c_str(), EN[ei], WorkingFamilies::name[wi], seed, aclamp, snf, enf);
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
    const WorkingFamilies families;
    const std::string fx = std::getenv("FS_TAG") ? std::string("_") + std::getenv("FS_TAG") : std::string();
    std::FILE* fo = open_result("famsw_seeds" + fx);
    std::FILE* fi = open_result("famsw_ident" + fx);
    std::FILE* fp = open_result("famsw_path" + fx);
    if (!fo || !fi || !fp) return missing_results_folder();
    // FS_H2 enables squared Hellinger at the Chapter 6 checkpoints.
    const bool fh2 = std::getenv("FS_H2") != nullptr;
    std::fprintf(fo, fh2 ? "gen,seed,method,window,rmse,excess_x1000,h2_x1000\n" : "gen,seed,method,window,rmse,excess_x1000\n");
    FamilyHellinger hellinger(h2, families.inverse, fh2);
    std::fprintf(fi, "gen,seed,kind,target,t50,t90,share_ge50_after,incorrect_majority,first_incorrect,pre_weight_target,n_guard\n");
    std::fprintf(fp, "gen,t,family,mean,sd\n");
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : thesis::kFirstSeed;
    // FS_EST = rmpl (default) | xiao | hermite: the five arms of that estimator (counting matrix / Hermite read the raw
    // data with their own margins, as in Sections 6.3-6.4; RMPL reads probit of the KLL pseudo-observations).
    const std::string fest = std::getenv("FS_EST") ? std::getenv("FS_EST") : "rmpl";
    std::vector<int> gens;
    if (const char* gs = std::getenv("FS_GENS")) {   // comma-separated subset of GN
        for (const auto& name : parse_list(gs))
            for (int g = 0; g < 7; ++g) if (name == GN[g]) gens.push_back(g);
    } else for (int g = 0; g < 7; ++g) gens.push_back(g);
    for (int g : gens) {
        const bool sw = G0[g] != G1[g];
        std::vector<double> ps(4 * kN, 0.0), pq(4 * kN, 0.0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const StudySeries s = load_study_series(GN[g], "constant", seed, kN);
            FixedMemoryArms arms(fest);
            // instance 0: family (20 models, index wi * 5 + k); instance 1: Gaussian-only (5 models)
            thesis::Mff family_mix(copula::Family::Gaussian, 4 * kK), gaussian_mix(copula::Family::Gaussian, kK);
            std::vector<double> sef(kN), seg(kN), exf(kN), exg(kN), hhf(kN, NAN), hhg(kN, NAN);
            std::vector<std::array<double, 4>> fw(kN);
            long nguard = 0;
            double th[4][kK], ld[4][kK], f[kK], family_forecasts[4 * kK];
            std::vector<double> family_losses(family_mix.size()), gaussian_losses(gaussian_mix.size());
            for (std::size_t t = 0; t < kN; ++t) {
                arms.forecasts(f);
                const bool ready = t >= thesis::kMffWarmup && arms.ready();
                const int gt = t < kChange ? G0[g] : G1[g];
                const auto truth_family = WorkingFamilies::family[gt];
                const double part = copula::param_from_spearman(truth_family, thesis::kConstantRhoS);
                const double lc = copula::log_density(truth_family, s.true_pit_x[t], s.true_pit_y[t], part);
                for (int wi = 0; wi < 4; ++wi)
                    for (std::size_t k = 0; k < kK; ++k) {
                        th[wi][k] = families.inverse[wi](f[k]);
                        ld[wi][k] = copula::log_density(WorkingFamilies::family[wi], s.true_pit_x[t], s.true_pit_y[t],
                                                        families.evaluation_parameter(wi, f[k]));
                    }
                // forecasts held before tick t; family arm i is working family i / kK at memory i % kK
                for (std::size_t i = 0; i < 4 * kK; ++i) family_forecasts[i] = f[i % kK];
                const double rf = family_mix.forecast(family_forecasts), rg = gaussian_mix.forecast(f);
                sef[t] = (rf - s.true_spearman[t]) * (rf - s.true_spearman[t]); seg[t] = (rg - s.true_spearman[t]) * (rg - s.true_spearman[t]);
                exf[t] = lc - family_mix.mixture_log_density(&ld[0][0]); exg[t] = lc - gaussian_mix.mixture_log_density(ld[0]);
                if (fh2 && checkpoint(sw ? "jump" : "constant", t)) {
                    int wis[20]; double rhs[20];
                    for (int i = 0; i < 20; ++i) { wis[i] = i / int(kK); rhs[i] = f[i % kK]; }
                    hhf[t] = hellinger.mixed_families(truth_family, part, wis, rhs, family_mix.w.data(), family_mix.size());
                    hhg[t] = hellinger.mixed_families(truth_family, part, wis, rhs, gaussian_mix.w.data(), gaussian_mix.size());
                }
                for (int wi = 0; wi < 4; ++wi) { double a = 0; for (std::size_t k = 0; k < kK; ++k) a += family_mix.w[wi * kK + k]; fw[t][wi] = a; }
                if (ready) {
                    for (int wi = 0; wi < 4; ++wi) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            family_losses[wi * kK + k] = -copula::log_density(WorkingFamilies::family[wi], s.estimated_pit_x[t],
                                                                              s.estimated_pit_y[t], th[wi][k]);
                        }
                    }
                    std::copy_n(family_losses.begin(), kK, gaussian_losses.begin());
                    nguard += thesis::guard_mff_losses(family_losses);
                    nguard += thesis::guard_mff_losses(gaussian_losses);
                    family_mix.update(family_losses);
                    gaussian_mix.update(gaussian_losses);
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
            std::fprintf(fi, "%s,%u,%s,%s,%ld,%ld,%.6f,%d,%ld,%.6f,%ld\n", GN[g], seed, sw ? "switch" : "initial", WorkingFamilies::name[tgt], t50, t90,
                         after ? double(ge) / after : NAN, int(inc), firstinc, sw ? fw[kChange - 1][tgt] : fw[0][tgt], nguard);
            for (std::size_t t = 0; t < kN; ++t) for (int wi = 0; wi < 4; ++wi) { ps[wi * kN + t] += fw[t][wi]; pq[wi * kN + t] += fw[t][wi] * fw[t][wi]; }
        }
        for (std::size_t t = 0; t < kN; t += 10)
            for (int wi = 0; wi < 4; ++wi) {
                const auto [mean, sd] = seed_mean_sd(ps[wi * kN + t], pq[wi * kN + t], seeds);
                std::fprintf(fp, "%s,%zu,%s,%.6f,%.6f\n", GN[g], t, WorkingFamilies::name[wi], mean, sd);
            }
    }
    std::fclose(fo); std::fclose(fi); std::fclose(fp);
    return 0;
}

}  // namespace strategy_experiments
