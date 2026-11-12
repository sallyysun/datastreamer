// Chapter 6 data for thesis Algorithm 1 (ForgettingBank): for Xiao, Hermite and online-MPL
// components, the five fixed memories, the squared-loss bank (tuned eta = 1, and the theory's
// eta = 1/32 as bank_sq32), the log-loss bank and the uniform-weight control, plus each component's own variable forgetting factor (VFF) at a slow
// and a fast rate, all fed the same shared rank pseudo-observations.
// Metrics per tick: squared error of rho against the true rho_S, and excess log loss at the true
// PIT, log c_true - log c_{g^{-1}(rho_hat)} (Gaussian method-of-moments density, section 4.3.1).
// Plan and caveats: notes/thesis/chapter6_plan.md.
//
// Usage (from experiments/):  ./probe_thesis_bank [seeds=20]   (COPULA=gaussian|clayton|gumbel|t)
//   VFF columns: the common VFF rule (vff.hpp) at rates 0.02 (slow) and 0.1 (fast).
//   COMPONENTS=mle,mlea,mleold TAG=startup  runs only those components into results/thesis_bank_startup[_fam].csv
//   -> results/thesis_bank[_<family>].csv          seed x scenario x window x method
//      results/thesis_bank_weights[_<family>].csv  mean bank weights per tick, jump scenario

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <limits>
#include <vector>

#include "../src/estimators/dependence/vff.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "density_metrics.hpp"
#include "scenarios.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
const unsigned kBaseSeed = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : 42u;
const std::vector<int> kMem{39, 99, 199, 399, 1999};
std::vector<std::string> kFamilies{"xiao", "hermite", "mle"};   // COMPONENTS=a,b,... overrides

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return scenario_drift(kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return scenario_oscillate();
}

copula::Family family_of(const std::string& f) {
    if (f == "clayton") return copula::Family::Clayton;
    if (f == "gumbel") return copula::Family::Gumbel;
    if (f == "t") return copula::Family::StudentT;
    return copula::Family::Gaussian;
}

// VFF comparators: the common rule (vff.hpp) wrapping each estimator, at the two controller rates
// fixed on development seeds 1-30 (ew_marginals.md §12). Same shared margins and training inputs as
// the bank arms; gregory has no MemoryControlled hook, so its rows are NaN.
constexpr double kVffSlow = 0.02, kVffFast = 0.1;
std::unique_ptr<DependenceEstimator> make_vff(const std::string& c, double rate) {
    VffController::Config v{.rate = rate};
    const bool raw = bank_experiment::raw();
    if (c == "xiao")
        return std::make_unique<CommonVff>(std::make_unique<XiaoSpearmanEstimator>(bank_experiment::xiao(0.99), "x"), v, raw, "vff");
    if (c == "hermite")
        return std::make_unique<CommonVff>(std::make_unique<HermiteEstimator>(bank_experiment::hermite(0.01), "h"), v, raw, "vff");
    if (c == "mle") {
        RecursiveMPL::Config m{};
        m.margin = RecursiveMPL::Margin::ExactNormal;
        m.gain = 0.01;
        bank_experiment::rmpl(m);
        return std::make_unique<CommonVff>(std::make_unique<RecursiveMPL>(m, "m"), v, false, "vff");
    }
    return nullptr;
}

double mom_logc(double rho, double u, double v) {
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}, {"early", kBurn, 2000}};
    if (s == "jump") {
        w.push_back({"pre", kBurn, kChange});
        w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500});
        w.push_back({"rec_500_1500", kChange + 500, kChange + 1500});
        w.push_back({"post", kChange, kN});
    }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 20;
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const copula::Family F = family_of(cop_name);
    const char* tag = std::getenv("TAG");   // output suffix, so a variant run doesn't overwrite the thesis CSVs
    const std::string sfx = (tag ? std::string("_") + tag : std::string()) + (cop_name == "gaussian" ? "" : "_" + cop_name);
    if (const char* comps = std::getenv("COMPONENTS")) {
        kFamilies.clear();
        std::string c(comps);
        for (std::size_t a = 0, b; a <= c.size(); a = b + 1) {
            b = c.find(',', a);
            if (b == std::string::npos) b = c.size();
            if (b > a) kFamilies.push_back(c.substr(a, b - a));
        }
    }

    // DENSITY_EVAL=1 (mle component only; chapter6_metrics.md §14): writes results/thesis_bank_density<sfx>.csv with, per
    // method, rho_S RMSE, excess log loss at the true PIT and squared Hellinger H^2(c_t, chat_{t|t-1}), for the fixed
    // arms, the no-forgetting MPL (the cumulative arm of mlecum_bank, on the same shared margins), each bank's single
    // Gaussian at its combined rho and its mixture sum_k w_k c_{r_k} (weights and arms held before tick t), and the
    // VFF. Evaluation only: the main CSV is unchanged. H^2 by normal-score quadrature (density_metrics.hpp, n = 81,
    // L = 8) every 50 ticks from 500, plus every 10 in [2500, 2600) on the jump path, counted only in rec_0_100.
    const bool dens_eval = std::getenv("DENSITY_EVAL") != nullptr;
    const dmetric::Grid hgrid = dmetric::Grid::full(81, 8.0);
    std::vector<std::string> scen_list{"constant", "drift", "jump", "oscillate"};
    if (const char* only = std::getenv("TB_SCENARIOS")) {   // a subset of the same scenarios
        std::vector<std::string> keep;
        for (const auto& sc : scen_list) if (("," + std::string(only) + ",").find("," + sc + ",") != std::string::npos) keep.push_back(sc);
        scen_list = keep;
    }
    std::FILE* dout = dens_eval ? std::fopen(("results/thesis_bank_density" + sfx + ".csv").c_str(), "w") : nullptr;
    if (dout) std::fprintf(dout, "copula,scenario,seed,component,window,method,metric,value\n");

    std::vector<std::string> methods;
    for (int m : kMem) methods.push_back("fixed_" + std::to_string(m));
    for (const char* m : {"bank", "bank_log", "uniform", "vff_slow", "vff_fast", "bank_sq32"}) methods.push_back(m);
    const std::size_t M = methods.size(), K = kMem.size();

    std::FILE* out = std::fopen(("results/thesis_bank" + sfx + ".csv").c_str(), "w");
    std::FILE* wout = std::fopen(("results/thesis_bank_weights" + sfx + ".csv").c_str(), "w");
    if (!out || !wout) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,component,window,method,rmse,excess_x1000\n");
    std::fprintf(wout, "copula,component,bank,t,k,mem,weight\n");

    for (const std::string& sc : scen_list) {
        // [component][bank 0=sq,1=log][t][k] summed over seeds, jump only
        std::vector<std::vector<std::vector<double>>> wsum(
            kFamilies.size() * 2, std::vector<std::vector<double>>(kN, std::vector<double>(K, 0.0)));
        for (int s = 0; s < seeds; ++s) {
            for (std::size_t c = 0; c < kFamilies.size(); ++c) {
                auto dep = make_dep(sc);
                auto cop = dgp::make_copula(cop_name);
                CopulaSource src(*dep, *cop, kBaseSeed + s);
                auto e_sq = make_estimator(kFamilies[c] + "_bank");
                auto e_log = make_estimator(kFamilies[c] + "_bank_log");
                auto e_uni = make_estimator(kFamilies[c] + "_bank_uniform");
                auto* sq = dynamic_cast<ForgettingBank*>(e_sq.get());
                auto* lg = dynamic_cast<ForgettingBank*>(e_log.get());
                auto* un = dynamic_cast<ForgettingBank*>(e_uni.get());
                auto e_s32 = make_estimator(kFamilies[c] + "_bank_sq32");
                auto vs = make_vff(kFamilies[c], kVffSlow), vf = make_vff(kFamilies[c], kVffFast);
                const bool dens = dens_eval && kFamilies[c] == "mle";
                auto e_cum = dens ? make_estimator("mlecum_bank") : nullptr;   // arm K = the cumulative (no-forgetting) MPL
                auto* cum = dens ? dynamic_cast<ForgettingBank*>(e_cum.get()) : nullptr;
                auto* s32 = dynamic_cast<ForgettingBank*>(e_s32.get());
                // density methods: 0-4 fixed, 5 nocum, 6-9 single (bank, bank_log, uniform, bank_sq32), 10-13 the same
                // banks' mixtures, 14-15 vff_slow / vff_fast
                const std::vector<std::string> dm{"fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "nocum",
                    "bank", "bank_log", "uniform", "bank_sq32", "bank_mix", "bank_log_mix", "uniform_mix", "bank_sq32_mix",
                    "vff_slow", "vff_fast"};
                const std::size_t D = dm.size();
                const float kNaNf = std::numeric_limits<float>::quiet_NaN();
                std::vector<std::vector<float>> dse(D, std::vector<float>(dens ? kN : 0)), dex(D, std::vector<float>(dens ? kN : 0)),
                    dh(D, std::vector<float>(dens ? kN : 0, kNaNf));

                std::vector<std::vector<float>> se(M, std::vector<float>(kN)), ex(M, std::vector<float>(kN));
                std::vector<double> rho(M);
                for (std::size_t t = 0; t < kN; ++t) {
                    const auto smp = src.next();
                    const double u = special::normal_cdf(smp.x), v = special::normal_cdf(smp.y);
                    const double par = copula::param_from_spearman(F, dgp::detail::target_spearman(smp.rho));
                    const double lc = copula::log_density(F, u, v, par);
                    // Predictions held before this tick; the fixed arms are the squared bank's own.
                    for (std::size_t k = 0; k < K; ++k) rho[k] = sq->model(k).estimate();
                    rho[K] = sq->estimate();
                    rho[K + 1] = lg->estimate();
                    rho[K + 2] = un->estimate();
                    rho[K + 3] = vs ? vs->estimate() : NAN;
                    rho[K + 4] = vf ? vf->estimate() : NAN;
                    rho[K + 5] = e_s32->estimate();
                    for (std::size_t i = 0; i < M; ++i) {
                        se[i][t] = float((rho[i] - smp.rho_s) * (rho[i] - smp.rho_s));
                        ex[i][t] = float(lc - mom_logc(rho[i], u, v));
                    }
                    if (dens) {
                        // Forecasts held before tick t: rho per method, and for the mixtures the bank's weights and arms.
                        const ForgettingBank* banks[4] = {sq, lg, un, s32};
                        const double rcum = cum->model(K).estimate();
                        std::vector<double> drho(D, NAN);
                        for (std::size_t k = 0; k < K; ++k) drho[k] = rho[k];
                        drho[5] = rcum;
                        drho[6] = rho[K]; drho[7] = rho[K + 1]; drho[8] = rho[K + 2]; drho[9] = rho[K + 5];
                        drho[14] = rho[K + 3]; drho[15] = rho[K + 4];
                        auto gpar = [](double r) { return copula::param_from_spearman(copula::Family::Gaussian, r); };
                        auto mixdens = [&](const ForgettingBank* b, double uu, double vv) {
                            double sum = 0;
                            for (std::size_t k = 0; k < K; ++k)
                                sum += b->weight(k) * std::exp(copula::Gaussian::log_density(uu, vv, gpar(b->model(k).estimate())));
                            return sum;
                        };
                        for (std::size_t i = 0; i < D; ++i) {
                            if (i >= 10 && i <= 13) {
                                dse[i][t] = se[K + (i == 13 ? 5 : i - 10)][t];   // a mixture shares its bank's rho
                                dex[i][t] = float(lc - std::log(mixdens(banks[i - 10], u, v)));
                            } else {
                                dse[i][t] = float((drho[i] - smp.rho_s) * (drho[i] - smp.rho_s));
                                dex[i][t] = float(lc - mom_logc(drho[i], u, v));
                            }
                        }
                        if (t >= kBurn && ((t - kBurn) % 50 == 0 || (sc == "jump" && t >= kChange && t < kChange + 100 && (t - kChange) % 10 == 0))) {
                            const dmetric::Table truth(hgrid, [&](double uu, double vv) {
                                const double l = copula::log_density(F, uu, vv, par);
                                return std::isfinite(l) ? std::exp(l) : 0.0;
                            });
                            for (std::size_t i = 0; i < D; ++i) {
                                const dmetric::Table f(hgrid, [&](double uu, double vv) {
                                    return (i >= 10 && i <= 13) ? mixdens(banks[i - 10], uu, vv)
                                                                : std::exp(copula::Gaussian::log_density(uu, vv, gpar(drho[i])));
                                });
                                dh[i][t] = float(dmetric::hellinger2(hgrid, truth, f));
                            }
                        }
                    }
                    sq->update(smp.x, smp.y);
                    lg->update(smp.x, smp.y);
                    un->update(smp.x, smp.y);
                    if (vs) vs->update(smp.x, smp.y);
                    if (vf) vf->update(smp.x, smp.y);
                    e_s32->update(smp.x, smp.y);
                    if (cum) cum->update(smp.x, smp.y);
                    if (sc == "jump")
                        for (std::size_t k = 0; k < K; ++k) {
                            wsum[2 * c][t][k] += sq->weight(k);
                            wsum[2 * c + 1][t][k] += lg->weight(k);
                        }
                }
                if (dens)
                    for (const auto& w : windows(sc)) {
                        const double n = double(w.b - w.a);
                        const bool dense = std::string(w.name) == "rec_0_100";
                        for (std::size_t i = 0; i < D; ++i) {
                            double a = 0, b = 0, h = 0;
                            int nh = 0;
                            for (std::size_t t = w.a; t < w.b; ++t) {
                                a += dse[i][t];
                                b += dex[i][t];
                                if (!std::isnan(dh[i][t]) && (dense || (t - kBurn) % 50 == 0)) { h += dh[i][t]; ++nh; }
                            }
                            const char* cn = cop_name.c_str();
                            std::fprintf(dout, "%s,%s,%u,mle,%s,%s,rmse,%.6f\n", cn, sc.c_str(), kBaseSeed + s, w.name, dm[i].c_str(), std::sqrt(a / n));
                            std::fprintf(dout, "%s,%s,%u,mle,%s,%s,excess_x1000,%.5f\n", cn, sc.c_str(), kBaseSeed + s, w.name, dm[i].c_str(), 1000 * b / n);
                            std::fprintf(dout, "%s,%s,%u,mle,%s,%s,hellinger2_x1000,%.6f\n", cn, sc.c_str(), kBaseSeed + s, w.name, dm[i].c_str(),
                                         nh ? 1000 * h / nh : NAN);
                        }
                    }
                for (const auto& w : windows(sc))
                    for (std::size_t i = 0; i < M; ++i) {
                        double a = 0, b = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) { a += se[i][t]; b += ex[i][t]; }
                        const double n = double(w.b - w.a);
                        std::fprintf(out, "%s,%s,%u,%s,%s,%s,%.6f,%.5f\n", cop_name.c_str(), sc.c_str(),
                                     kBaseSeed + s, kFamilies[c].c_str(), w.name, methods[i].c_str(),
                                     std::sqrt(a / n), 1000 * b / n);
                    }
            }
        }
        if (sc == "jump")
            for (std::size_t c = 0; c < kFamilies.size(); ++c)
                for (int b = 0; b < 2; ++b)
                    for (std::size_t t = 2000; t < 4000; t += 5)
                        for (std::size_t k = 0; k < K; ++k)
                            std::fprintf(wout, "%s,%s,%s,%zu,%zu,%d,%.5f\n", cop_name.c_str(),
                                         kFamilies[c].c_str(), b ? "bank_log" : "bank", t, k, kMem[k],
                                         wsum[2 * c + b][t][k] / seeds);
        std::fprintf(stderr, "%s done\n", sc.c_str());
    }
    std::fclose(out);
    std::fclose(wout);
    if (dout) std::fclose(dout);
    return 0;
}
