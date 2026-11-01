// Chapter 6 data for thesis Algorithm 1 (ForgettingBank): for Xiao, Hermite and online-MPL
// components, the five fixed memories, the squared-loss bank, the log-loss bank and the
// uniform-weight control, plus each component's own variable forgetting factor (VFF) at a slow
// and a fast rate, all fed the same shared rank pseudo-observations.
// Metrics per tick: squared error of rho against the true rho_S, and excess log loss at the true
// PIT, log c_true - log c_{g^{-1}(rho_hat)} (Gaussian method-of-moments density, section 4.3.1).
// Plan and caveats: notes/thesis/chapter6_plan.md.
//
// Usage (from experiments/):  ./probe_thesis_bank [seeds=20]   (COPULA=gaussian|clayton|gumbel|t)
//   -> results/thesis_bank[_<family>].csv          seed x scenario x window x method
//      results/thesis_bank_weights[_<family>].csv  mean bank weights per tick, jump scenario

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
constexpr unsigned kBaseSeed = 42;
const std::vector<int> kMem{39, 99, 199, 399, 1999};
const std::vector<std::string> kFamilies{"xiao", "hermite", "mle"};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}

copula::Family family_of(const std::string& f) {
    if (f == "clayton") return copula::Family::Clayton;
    if (f == "gumbel") return copula::Family::Gumbel;
    if (f == "t") return copula::Family::StudentT;
    return copula::Family::Gaussian;
}

// A VFF estimator fed the bank's inputs: probit of a 500-tick rank PIT, as ForgettingBank does.
struct PitFed {
    std::unique_ptr<DependenceEstimator> e;
    RankPit px{500}, py{500};
    void update(double x, double y) {
        const double u = px.transform(x), v = py.transform(y);
        e->update(RankPit::probit(u), RankPit::probit(v));
    }
};

// Every VFF starts at M = 199 ticks (the bank's middle arm) and is clamped to the bank's span
// where its rule allows it (Xiao); Hermite's clamp is fixed at [0.2, 20] x lambda.
std::unique_ptr<DependenceEstimator> make_vff(const std::string& c, double rate) {
    if (c == "xiao")
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.99, .calibration = 100,
                                          .vff_rate = rate, .vff_mem_min = 39, .vff_mem_max = 1999}, "vff");
    if (c == "hermite")
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.01, .burn_in = 50, .vff_rate = rate}, "vff");
    return std::make_unique<ParametricCopulaMLE>(
        ParametricCopulaMLE::Config{.family = copula::Family::Gaussian, .lambda = 0.99, .vff_rate = rate,
                                    .expected_info = true, .burn_in = 50}, "vff");
}

double mom_logc(double rho, double u, double v) {
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
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
    const std::string sfx = cop_name == "gaussian" ? "" : "_" + cop_name;

    std::vector<std::string> methods;
    for (int m : kMem) methods.push_back("fixed_" + std::to_string(m));
    for (const char* m : {"bank", "bank_log", "uniform", "vff_slow", "vff_fast"}) methods.push_back(m);
    const std::size_t M = methods.size(), K = kMem.size();

    std::FILE* out = std::fopen(("results/thesis_bank" + sfx + ".csv").c_str(), "w");
    std::FILE* wout = std::fopen(("results/thesis_bank_weights" + sfx + ".csv").c_str(), "w");
    if (!out || !wout) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,component,window,method,rmse,excess_x1000\n");
    std::fprintf(wout, "copula,component,bank,t,k,mem,weight\n");

    for (const std::string sc : {"constant", "drift", "jump", "oscillate"}) {
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
                PitFed vs{make_vff(kFamilies[c], 0.05)}, vf{make_vff(kFamilies[c], 0.5)};

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
                    rho[K + 3] = vs.e->estimate();
                    rho[K + 4] = vf.e->estimate();
                    for (std::size_t i = 0; i < M; ++i) {
                        se[i][t] = float((rho[i] - smp.rho_s) * (rho[i] - smp.rho_s));
                        ex[i][t] = float(lc - mom_logc(rho[i], u, v));
                    }
                    sq->update(smp.x, smp.y);
                    lg->update(smp.x, smp.y);
                    un->update(smp.x, smp.y);
                    vs.update(smp.x, smp.y);
                    vf.update(smp.x, smp.y);
                    if (sc == "jump")
                        for (std::size_t k = 0; k < K; ++k) {
                            wsum[2 * c][t][k] += sq->weight(k);
                            wsum[2 * c + 1][t][k] += lg->weight(k);
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
    return 0;
}
