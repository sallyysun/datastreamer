// Tuning eta and delta for thesis Algorithm 1 (ForgettingBank) on the drift and jump scenarios.
// The arms and the shared rank PIT do not depend on (eta, delta), so each seed runs them once and
// replays the Hedge + fixed-share recursion for every grid point; this reproduces ForgettingBank
// exactly (checked against probe_thesis_bank at the registry's eta, delta).
// Protocol and results: notes/studies/thesis_bank_tuning.md.
//
// Usage (from experiments/):  ./probe_bank_tune <first_seed> <seeds>   (COPULA=gaussian|clayton|gumbel|t)
//   -> results/bank_tune_<first_seed>[_<family>].csv   seed x scenario x component x window x method

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kWarm = 30, kPit = 500;
const std::vector<int> kMem{39, 99, 199, 399, 1999};
const std::vector<std::string> kComponents{"xiao", "hermite", "mle"};
const std::vector<double> kEtaSq{0.1, 0.25, 0.5, 1, 2, 4, 8, 16};
const std::vector<double> kEtaLog{0.25, 0.5, 1, 2, 4, 8, 16};
const std::vector<double> kDelta{0, 1e-5, 3e-5, 1e-4, 3e-4, 4e-4, 1e-3, 3e-3, 1e-2};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
}

copula::Family family_of(const std::string& f) {
    if (f == "clayton") return copula::Family::Clayton;
    if (f == "gumbel") return copula::Family::Gumbel;
    if (f == "t") return copula::Family::StudentT;
    return copula::Family::Gaussian;
}

double mom_logc(double rho, double u, double v) {
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") {
        w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500});
        w.push_back({"rec_500_1500", kChange + 500, kChange + 1500});
        w.push_back({"post", kChange, kN});
    }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_bank_tune <first_seed> <seeds>\n"); return 1; }
    const unsigned first = static_cast<unsigned>(std::atoi(argv[1]));
    const int seeds = std::atoi(argv[2]);
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const copula::Family F = family_of(cop_name);
    const std::string sfx = cop_name == "gaussian" ? "" : "_" + cop_name;
    const std::size_t K = kMem.size();

    struct Cfg { std::string name; bool log; double eta, delta; };
    std::vector<Cfg> cfgs;
    for (double e : kEtaSq) for (double d : kDelta) cfgs.push_back({"sq", false, e, d});
    for (double e : kEtaLog) for (double d : kDelta) cfgs.push_back({"log", true, e, d});

    const std::string path = "results/bank_tune_" + std::to_string(first) + sfx + ".csv";
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,component,window,method,eta,delta,rmse,excess_x1000\n");

    std::vector<std::vector<double>> arm(K, std::vector<double>(kN)), lsq(K, std::vector<double>(kN)),
        llog(K, std::vector<double>(kN));
    std::vector<double> truth(kN), lc(kN), uu(kN), vv(kN);
    std::vector<char> scorable(kN);
    std::vector<double> se(kN), ex(kN), w(K), lw(K);

    for (const std::string sc : {"drift", "jump"}) {
        for (int s = 0; s < seeds; ++s) {
            for (const auto& comp : kComponents) {
                auto dep = make_dep(sc);
                auto cop = dgp::make_copula(cop_name);
                CopulaSource src(*dep, *cop, first + s);
                auto e = make_estimator(comp + "_bank_uniform");
                auto* bank = dynamic_cast<ForgettingBank*>(e.get());
                RankPit px{kPit}, py{kPit};
                for (std::size_t t = 0; t < kN; ++t) {
                    const auto smp = src.next();
                    const double u0 = special::normal_cdf(smp.x), v0 = special::normal_cdf(smp.y);
                    truth[t] = smp.rho_s;
                    lc[t] = copula::log_density(F, u0, v0,
                                                copula::param_from_spearman(F, dgp::detail::target_spearman(smp.rho)));
                    uu[t] = u0;
                    vv[t] = v0;
                    // The bank's own margins, replayed: scorable once the PIT holds kWarm values.
                    scorable[t] = px.size() >= kWarm;
                    const double u = px.transform(smp.x), v = py.transform(smp.y);
                    const double p = 12.0 * (u - 0.5) * (v - 0.5);
                    for (std::size_t k = 0; k < K; ++k) {
                        const double r = bank->model(k).estimate();
                        arm[k][t] = r;
                        lsq[k][t] = (p - r) * (p - r);
                        llog[k][t] = -copula::Gaussian::log_density(
                            u, v, copula::param_from_spearman(copula::Family::Gaussian, r));
                    }
                    bank->update(smp.x, smp.y);
                }
                const auto wins = windows(sc);
                auto emit = [&](const std::string& m, double eta, double delta) {
                    for (const auto& wn : wins) {
                        double a = 0, b = 0;
                        for (std::size_t t = wn.a; t < wn.b; ++t) { a += se[t]; b += ex[t]; }
                        const double n = double(wn.b - wn.a);
                        std::fprintf(out, "%s,%s,%u,%s,%s,%s,%g,%g,%.6f,%.5f\n", cop_name.c_str(), sc.c_str(),
                                     first + s, comp.c_str(), wn.name, m.c_str(), eta, delta, std::sqrt(a / n),
                                     1000 * b / n);
                    }
                };
                for (std::size_t k = 0; k < K; ++k) {
                    for (std::size_t t = 0; t < kN; ++t) {
                        se[t] = (arm[k][t] - truth[t]) * (arm[k][t] - truth[t]);
                        ex[t] = lc[t] - mom_logc(arm[k][t], uu[t], vv[t]);
                    }
                    emit("fixed_" + std::to_string(kMem[k]), 0, 0);
                }
                for (const auto& c : cfgs) {
                    const auto& L = c.log ? llog : lsq;
                    std::fill(w.begin(), w.end(), 1.0 / double(K));
                    for (std::size_t t = 0; t < kN; ++t) {
                        double r = 0;
                        for (std::size_t k = 0; k < K; ++k) r += w[k] * arm[k][t];
                        r = std::clamp(r, -1.0, 1.0);
                        se[t] = (r - truth[t]) * (r - truth[t]);
                        ex[t] = lc[t] - mom_logc(r, uu[t], vv[t]);
                        if (!scorable[t]) continue;
                        double worst = 0, top = -std::numeric_limits<double>::infinity();
                        for (std::size_t k = 0; k < K; ++k) if (std::isfinite(L[k][t])) worst = std::max(worst, L[k][t]);
                        for (std::size_t k = 0; k < K; ++k) {
                            const double l = std::isfinite(L[k][t]) ? L[k][t] : worst;
                            lw[k] = (w[k] > 0 ? std::log(w[k]) : -std::numeric_limits<double>::infinity()) - c.eta * l;
                            top = std::max(top, lw[k]);
                        }
                        double total = 0;
                        for (std::size_t k = 0; k < K; ++k) { w[k] = std::exp(lw[k] - top); total += w[k]; }
                        for (std::size_t k = 0; k < K; ++k) w[k] = (1.0 - c.delta) * w[k] / total + c.delta / double(K);
                    }
                    emit(c.name, c.eta, c.delta);
                }
            }
        }
        std::fprintf(stderr, "%s %s done\n", cop_name.c_str(), sc.c_str());
    }
    std::fclose(out);
    return 0;
}
