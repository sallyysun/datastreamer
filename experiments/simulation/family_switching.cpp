// Family identification and adaptation at matched Spearman dependence.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "../../src/special_functions.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

using thesis::H2Table;
using thesis::Win;
using thesis::checkpoint;
using thesis::kBurn;
using thesis::kChange;
using thesis::kK;
using thesis::kMem;
using thesis::kN;

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
    // FS_H2=1: also squared Hellinger H^2(c_true, mixture) at the Chapter 6 checkpoints (every 50 from tick 500; every
    // 10 in [2500, 2600) on switch paths, counted only in rec_0_100), on the 81 x 81 grid of the 6.3 evaluation:
    // Gaussian components in closed form (as H2Table::mix), t4 / Clayton / Gumbel components tabulated on a rho_S grid
    // (step 0.0025) and interpolated (checked against exact integration at start). Truth table per (family, parameter).
    const bool fh2 = std::getenv("FS_H2") != nullptr;
    std::fprintf(fo, fh2 ? "gen,seed,method,window,rmse,excess_x1000,h2_x1000\n" : "gen,seed,method,window,rmse,excess_x1000\n");
    const int ng = h2.g.size(); const int nr = 799; const double rlo = -0.995, rst = 0.0025;
    std::vector<std::vector<float>> ctab(4);
    std::map<std::pair<int, double>, std::vector<double>> truthc;
    if (fh2)
        for (int wi = 1; wi < 4; ++wi) {
            ctab[wi].resize(std::size_t(nr) * ng * ng);
            for (int a = 0; a < nr; ++a) {
                const double thv = theta_of(wi, rlo + rst * a);
                for (int i = 0; i < ng; ++i)
                    for (int j = 0; j < ng; ++j) {
                        const double l = copula::log_density(WF[wi], h2.g.u[i], h2.g.u[j], thv);
                        ctab[wi][(std::size_t(a) * ng + i) * ng + j] = float(std::isfinite(l) ? std::exp(l) : 0.0);
                    }
            }
        }
    // H^2 of the truth (family gt, parameter par) against sum_m w_m c_{wi_m}(rho_m)
    auto h2mix = [&](int gt, double par, const int* wis, const double* rhos, const double* ws, int nm) {
        auto key = std::make_pair(gt, par);
        auto it = truthc.find(key);
        if (it == truthc.end()) {
            const dmetric::Table tt(h2.g, [&](double a, double b) { const double l = copula::log_density(WF[gt], a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            it = truthc.emplace(key, tt.v).first;
        }
        std::vector<double> r(nm), c0(nm), c1(nm), fr(nm); std::vector<std::size_t> i0(nm);
        for (int m = 0; m < nm; ++m) {
            if (wis[m] == 0) {
                r[m] = copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rhos[m], -0.999, 0.999));
                c0[m] = 1 / std::sqrt(1 - r[m] * r[m]); c1[m] = 1 / (2 * (1 - r[m] * r[m]));
            } else {
                const double x = (std::clamp(rhos[m], rlo, rlo + rst * (nr - 1)) - rlo) / rst;
                i0[m] = std::min(std::size_t(nr - 2), std::size_t(x)); fr[m] = x - i0[m];
            }
        }
        double sacc = 0;
        for (int i = 0; i < ng; ++i)
            for (int j = 0; j < ng; ++j) {
                const double x = h2.g.z[i], y = h2.g.z[j];
                const std::size_t o = std::size_t(i) * ng + j;
                double mm = 0;
                for (int m = 0; m < nm; ++m) {
                    if (ws[m] <= 0) continue;
                    if (wis[m] == 0) mm += ws[m] * c0[m] * std::exp(-(r[m] * r[m] * (x * x + y * y) - 2 * r[m] * x * y) * c1[m]);
                    else { const float* T0 = &ctab[wis[m]][i0[m] * ng * ng]; mm += ws[m] * ((1 - fr[m]) * T0[o] + fr[m] * T0[o + ng * ng]); }
                }
                sacc += std::sqrt(std::max(it->second[o], 0.0) * mm) * h2.g.w[i] * h2.g.w[j];
            }
        return 1.0 - sacc;
    };
    if (fh2) {   // tabulated components vs exact dmetric integration, one mixture per non-Gaussian family
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi) {
            const int wis[3] = {wi, wi, 0}; const double rhs[3] = {0.31, 0.6123, 0.55}, wss[3] = {0.3, 0.5, 0.2};
            const double par = copula::param_from_spearman(WF[wi], thesis::kConstantRhoS);
            const dmetric::Table tt(h2.g, [&](double a, double b) { const double l = copula::log_density(WF[wi], a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 3; ++k) { const double th = wis[k] == 0 ? copula::param_from_spearman(copula::Family::Gaussian, rhs[k]) : copula::param_from_spearman(WF[wis[k]], rhs[k]); const double l = copula::log_density(WF[wis[k]], a, b, th); m += wss[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
            const double ex = dmetric::hellinger2(h2.g, tt, mt), tb = h2mix(wi, par, wis, rhs, wss, 3);
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
            const Stream s = make_stream(GN[g], "constant", seed);
            std::vector<std::unique_ptr<DependenceEstimator>> arms;
            for (double n : kMem) arms.push_back(arm(fest, n));
            // instance 0: family (20 models, index wi * 5 + k); instance 1: Gaussian-only (5 models)
            std::vector<double> wf(20, 1.0 / 20), wg(kK, 1.0 / kK);
            std::vector<double> sef(kN), seg(kN), exf(kN), exg(kN), hhf(kN, NAN), hhg(kN, NAN);
            std::vector<std::array<double, 4>> fw(kN);
            long nguard = 0;
            double th[4][kK], te[4][kK], ld[4][kK], f[kK];
            std::vector<double> family_losses(wf.size()), gaussian_losses(wg.size()), log_weights(wf.size());
            for (std::size_t t = 0; t < kN; ++t) {
                bool ready = t >= 30;
                for (std::size_t k = 0; k < kK; ++k) { f[k] = arms[k]->estimate(); ready = ready && arms[k]->ready(); }
                const int gt = t < kChange ? G0[g] : G1[g];
                const double part = copula::param_from_spearman(WF[gt], thesis::kConstantRhoS);
                const double lc = copula::log_density(WF[gt], s.pu[t], s.pv[t], part);
                for (int wi = 0; wi < 4; ++wi)
                    for (std::size_t k = 0; k < kK; ++k) {
                        th[wi][k] = theta_of(wi, f[k]);
                        te[wi][k] = wi == 0 ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(f[k], -0.999, 0.999)) : th[wi][k];
                        ld[wi][k] = copula::log_density(WF[wi], s.pu[t], s.pv[t], te[wi][k]);
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
                sef[t] = (rf - s.rho[t]) * (rf - s.rho[t]); seg[t] = (rg - s.rho[t]) * (rg - s.rho[t]);
                exf[t] = lc - mixlog(wf.data(), 20, true); exg[t] = lc - mixlog(wg.data(), kK, false);
                if (fh2 && checkpoint(sw ? "jump" : "constant", t)) {
                    int wis[20]; double rhs[20];
                    for (int i = 0; i < 20; ++i) { wis[i] = i / int(kK); rhs[i] = f[i % kK]; }
                    hhf[t] = h2mix(gt, part, wis, rhs, wf.data(), 20);
                    hhg[t] = h2mix(gt, part, wis, rhs, wg.data(), int(kK));   // first five entries: the Gaussian models
                }
                for (int wi = 0; wi < 4; ++wi) { double a = 0; for (std::size_t k = 0; k < kK; ++k) a += wf[wi * kK + k]; fw[t][wi] = a; }
                if (ready) {
                    for (int wi = 0; wi < 4; ++wi) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            family_losses[wi * kK + k] = -copula::log_density(WF[wi], s.u[t], s.v[t], th[wi][k]);
                        }
                    }
                    std::copy_n(family_losses.begin(), kK, gaussian_losses.begin());
                    nguard += thesis::guard_mff_losses(family_losses);
                    nguard += thesis::guard_mff_losses(gaussian_losses);
                    thesis::apply_fixed_share(wf, family_losses, log_weights, 1.0, 1e-4);
                    thesis::apply_fixed_share(wg, gaussian_losses, log_weights, 1.0, 1e-4);
                }
                const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                for (auto& a : arms) { if (fest == "rmpl") a->update(zx, zy); else a->update(s.x[t], s.y[t]); }   // as in 6.3
            }
            // windows
            std::vector<Win> wins{{"whole", kBurn, kN}};
            if (sw) { wins.push_back({"pre", kBurn, kChange}); wins.push_back({"rec_0_100", kChange, kChange + 100});
                      wins.push_back({"late", kChange + 100, kN}); wins.push_back({"post", kChange, kN}); }
            for (const auto& w : wins)
                for (int m = 0; m < 2; ++m) {
                    double a = 0, b = 0, hsum = 0; int nh = 0;
                    const bool dense = std::string(w.name) == "rec_0_100";
                    for (std::size_t t = w.a; t < w.b; ++t) {
                        a += m ? seg[t] : sef[t]; b += m ? exg[t] : exf[t];
                        const double hv = m ? hhg[t] : hhf[t];
                        if (!std::isnan(hv) && (dense || (t - kBurn) % 50 == 0)) { hsum += hv; ++nh; }
                    }
                    const double n = double(w.b - w.a);
                    if (fh2) std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * hsum / nh : NAN);
                    else std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, std::sqrt(a / n), 1000 * b / n);
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
