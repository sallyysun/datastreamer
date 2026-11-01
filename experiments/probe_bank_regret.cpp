// Empirical check of the section 5.3.4 regret bounds for thesis Algorithm 1 (ForgettingBank).
// For each run it replays the bank's weights over the scored ticks and compares the bank's
// cumulative loss with the best single arm and the best arm sequence with at most m switches
// (dynamic programme), against B_{T,m}(delta) = log K + (T-1-m) log 1/(1-delta+delta/K) + m log K/delta.
// Banks: "log_mix"   log loss of the mixture of the arms' Gaussian MoM copula densities (eta=1;
//                    the forecaster the log-loss bound covers);
//        "log_point" log loss of the Gaussian MoM density at the bank's own rho (sum_k w_k rho_k),
//                    same weights (not covered: the density at the mean rho is not the mixture);
//        "sq"        squared loss (p_hat - sum_k w_k rho_k)^2 at the tuned eta=1 (not covered: needs
//                    eta <= 1/32);
//        "sq32"      squared loss at eta = 1/32, delta = 1e-4 (covered: bound 32 B);
//        "sq32_dstar" the same at the theory's delta* = K m/((K-1)(T-1)) for m = 1.
// With a third argument `family`, the component is FamilyBank (18 arms, shapes x memories) instead,
// scored by each arm's own copula log density: "log_mix" (eta = 1, delta* = 2.1e-4, the registry's) and
// "log_mix_hedge" (delta = 0). notes/studies/thesis_bank_family.md.
// Notes: notes/studies/thesis_bank_regret.md.
//
// Usage (from experiments/):  ./probe_bank_regret <first_seed> <seeds>   (COPULA=gaussian|clayton|gumbel|t)
//   -> results/bank_regret_<first_seed>[_<family>].csv
//      ./probe_bank_regret <first_seed> <seeds> family  -> results/bank_regret_family_<first_seed>[_<family>].csv

#include <algorithm>
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

constexpr std::size_t kN = 5000, kChange = 2500, kWarm = 30, kPit = 500;
constexpr double kEtaSq = 1.0, kDeltaSq = 1e-4, kEtaLog = 1.0, kDeltaLog = 3e-4;
const std::vector<std::string> kComponents{"xiao", "hermite", "mle"};
const std::vector<int> kSwitches{0, 1, 2, 3, 5, 10, 20, 50};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
}

double penalty(std::size_t T, int m, std::size_t K, double delta) {
    const double k = double(K);
    return std::log(k) + double(T - 1 - m) * -std::log1p(-delta + delta / k) + (m > 0 ? m * std::log(k / delta) : 0.0);
}

// Best cumulative loss over arm sequences with at most j switches, for every j in 0..jmax.
std::vector<double> best_sequences(const std::vector<std::vector<double>>& L, int jmax) {
    const std::size_t K = L.size(), T = L[0].size();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> D(jmax + 1, std::vector<double>(K, inf)), E = D;
    for (std::size_t k = 0; k < K; ++k) D[0][k] = L[k][0];
    for (std::size_t t = 1; t < T; ++t) {
        for (int j = 0; j <= jmax; ++j) {
            double m1 = inf, m2 = inf;
            std::size_t a1 = K;
            if (j > 0)
                for (std::size_t i = 0; i < K; ++i) {
                    if (D[j - 1][i] < m1) { m2 = m1; m1 = D[j - 1][i]; a1 = i; }
                    else if (D[j - 1][i] < m2) m2 = D[j - 1][i];
                }
            for (std::size_t k = 0; k < K; ++k) {
                const double sw = j > 0 ? (k == a1 ? m2 : m1) : inf;
                E[j][k] = L[k][t] + std::min(D[j][k], sw);
            }
        }
        std::swap(D, E);
    }
    std::vector<double> out(jmax + 1);
    for (int j = 0; j <= jmax; ++j) {
        double b = inf;
        for (int i = 0; i <= j; ++i) b = std::min(b, *std::min_element(D[i].begin(), D[i].end()));
        out[j] = b;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_bank_regret <first_seed> <seeds>\n"); return 1; }
    const unsigned first = static_cast<unsigned>(std::atoi(argv[1]));
    const int seeds = std::atoi(argv[2]);
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const std::string sfx = cop_name == "gaussian" ? "" : "_" + cop_name;
    const int jmax = kSwitches.back();
    const bool family = argc > 3 && std::string(argv[3]) == "family";
    const std::vector<std::string> components = family ? std::vector<std::string>{"family"} : kComponents;

    const std::string path = std::string("results/bank_regret_") + (family ? "family_" : "") + std::to_string(first) + sfx + ".csv";
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,component,bank,T,m,L_bank,L_best,regret,bound,best_arm_mem\n");

    for (const std::string sc : {"drift", "jump"}) {
        for (int s = 0; s < seeds; ++s) {
            for (const auto& comp : components) {
                auto dep = make_dep(sc);
                auto cop = dgp::make_copula(cop_name);
                CopulaSource src(*dep, *cop, first + s);
                auto e = make_estimator(comp + "_bank_uniform");
                auto* bank = dynamic_cast<ForgettingBank*>(e.get());
                auto* fbank = dynamic_cast<FamilyBank*>(e.get());
                const std::size_t K = bank ? bank->size() : fbank->size();
                std::vector<int> mems;  // arm memory, in the bank's arm order
                for (std::size_t k = 0; k < K; ++k) mems.push_back(bank ? std::vector<int>{39, 99, 199, 399, 1999}[k] : fbank->arm(k).mem);
                RankPit px{kPit}, py{kPit};
                // Per scored tick: arm rho, PIT pair, losses.
                std::vector<std::vector<double>> rho(K), lsq(K), llog(K);
                std::vector<double> pp, uu, vv;
                for (std::size_t t = 0; t < kN; ++t) {
                    const auto smp = src.next();
                    const bool scorable = px.size() >= kWarm;
                    const double u = px.transform(smp.x), v = py.transform(smp.y);
                    if (scorable) {
                        const double p = 12.0 * (u - 0.5) * (v - 0.5);
                        pp.push_back(p); uu.push_back(u); vv.push_back(v);
                        double worst = 0;
                        std::vector<double> lg(K);
                        for (std::size_t k = 0; k < K; ++k) {
                            const double r = bank ? bank->model(k).estimate() : fbank->arm_rho(k);
                            rho[k].push_back(r);
                            lsq[k].push_back((p - r) * (p - r));
                            lg[k] = bank ? -copula::Gaussian::log_density(
                                               u, v, copula::param_from_spearman(copula::Family::Gaussian, r))
                                         : -fbank->arm_log_density(k, u, v);
                            if (std::isfinite(lg[k])) worst = std::max(worst, lg[k]);
                        }
                        // Same guard as ForgettingBank: a non-finite loss becomes the tick's worst finite loss.
                        for (std::size_t k = 0; k < K; ++k) llog[k].push_back(std::isfinite(lg[k]) ? lg[k] : worst);
                    }
                    e->update(smp.x, smp.y);
                }
                const std::size_t T = pp.size();

                auto run = [&](const char* name, bool log, double eta, double delta) {
                    const auto& L = log ? llog : lsq;
                    std::vector<double> w(K, 1.0 / double(K)), lw(K);
                    double Lmix = 0, Lpt = 0;
                    for (std::size_t t = 0; t < T; ++t) {
                        double r = 0, top = -std::numeric_limits<double>::infinity();
                        for (std::size_t k = 0; k < K; ++k) r += w[k] * rho[k][t];
                        r = std::clamp(r, -1.0, 1.0);
                        if (log) {
                            for (std::size_t k = 0; k < K; ++k) {
                                lw[k] = std::log(w[k]) - L[k][t];
                                top = std::max(top, lw[k]);
                            }
                            double s = 0;
                            for (std::size_t k = 0; k < K; ++k) s += std::exp(lw[k] - top);
                            Lmix += -(top + std::log(s));
                            const double lp = -copula::Gaussian::log_density(
                                uu[t], vv[t], copula::param_from_spearman(copula::Family::Gaussian, r));
                            Lpt += std::isfinite(lp) ? lp : 0;
                        } else {
                            Lmix += (pp[t] - r) * (pp[t] - r);
                        }
                        top = -std::numeric_limits<double>::infinity();
                        for (std::size_t k = 0; k < K; ++k) {
                            lw[k] = std::log(w[k]) - eta * L[k][t];
                            top = std::max(top, lw[k]);
                        }
                        double total = 0;
                        for (std::size_t k = 0; k < K; ++k) { w[k] = std::exp(lw[k] - top); total += w[k]; }
                        for (std::size_t k = 0; k < K; ++k) w[k] = (1.0 - delta) * w[k] / total + delta / double(K);
                    }
                    const auto best = best_sequences(L, jmax);
                    std::size_t barm = 0;
                    double bsum = std::numeric_limits<double>::infinity();
                    for (std::size_t k = 0; k < K; ++k) {
                        double a = 0;
                        for (double x : L[k]) a += x;
                        if (a < bsum) { bsum = a; barm = k; }
                    }
                    auto emit = [&](const std::string& b, double Lb) {
                        for (int m : kSwitches)
                            std::fprintf(out, "%s,%s,%u,%s,%s,%zu,%d,%.4f,%.4f,%.4f,%.4f,%d\n", cop_name.c_str(),
                                         sc.c_str(), first + s, comp.c_str(), b.c_str(), T, m, Lb, best[m], Lb - best[m],
                                         penalty(T, m, K, delta) / eta, mems[barm]);
                    };
                    emit(name, Lmix);
                    if (log && delta > 0 && bank) emit("log_point", Lpt);
                };
                if (fbank) {
                    run("log_mix", true, 1.0, 2.1e-4);
                    run("log_mix_hedge", true, 1.0, 0.0);
                    continue;
                }
                run("log_mix", true, kEtaLog, kDeltaLog);
                run("sq", false, kEtaSq, kDeltaSq);
                run("log_mix_hedge", true, kEtaLog, 0.0);
                run("sq32", false, 1.0 / 32.0, kDeltaSq);
                run("sq32_dstar", false, 1.0 / 32.0, double(K) / (double(K - 1) * double(T - 1)));
            }
        }
        std::fprintf(stderr, "%s %s done\n", cop_name.c_str(), sc.c_str());
    }
    std::fclose(out);
    return 0;
}
