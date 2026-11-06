// Whole-copula accuracy of Xiao, Hermite and the spec MPL (budget_mpl.md §18). For each estimator:
// its own predictive density at each of the five bank memories (N_eff 39..1999), its own-density bank
// (mixture of the arms' densities, *_bank_own weights) and its log bank read as a Gaussian copula at
// the bank's rho. Two metrics, at checkpoints every 100 ticks on a G x G midpoint grid in u:
//   density ISE on the normal-score scale  int (f_hat - f)^2 dx dy = int (c_hat - c)^2 phi(z_u) phi(z_v) du dv
//   copula CDF ISE                          int (C_hat - C)^2 du dv  (CDFs by cumulating cell masses)
// Same streams as probe_thesis_bank (seeds 42-101). Shape switches: COPULA=clayton2gumbel|gauss2t with the
// `shape` scenario (rho_S fixed, family switches at 2500).
// Usage (from experiments/): ./probe_ise [seeds=60]  (COPULA=...)  -> results/ise[_fam].csv
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
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kEvery = 100, G = 32;
const int kNeff[5] = {39, 99, 199, 399, 1999};

struct Fam { copula::Family before, after; bool switches; };
Fam fam_of(const std::string& c) {
    using F = copula::Family;
    if (c == "clayton") return {F::Clayton, F::Clayton, false};
    if (c == "gumbel") return {F::Gumbel, F::Gumbel, false};
    if (c == "t") return {F::StudentT, F::StudentT, false};
    if (c == "clayton2gumbel") return {F::Clayton, F::Gumbel, true};
    if (c == "gauss2t") return {F::Gaussian, F::StudentT, true};
    return {F::Gaussian, F::Gaussian, false};
}

struct Grid {
    std::vector<double> u, w;   // midpoints and phi(probit(u))
    Grid() {
        for (std::size_t i = 0; i < G; ++i) {
            const double x = (i + 0.5) / G;
            u.push_back(x);
            const double z = RankPit::probit(x);
            w.push_back(std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI));
        }
    }
};

// density ISE (normal-score scale) and CDF ISE between two G x G copula-density grids
std::pair<double, double> ise(const std::vector<double>& c, const std::vector<double>& t, const Grid& g) {
    double sd = 0.0, sc = 0.0;
    std::vector<double> C(G * G, 0.0), T(G * G, 0.0);
    for (std::size_t i = 0; i < G; ++i)
        for (std::size_t j = 0; j < G; ++j) {
            const double e = c[i * G + j] - t[i * G + j];
            sd += e * e * g.w[i] * g.w[j];
            const double mc = c[i * G + j] / (G * G), mt = t[i * G + j] / (G * G);
            C[i * G + j] = mc + (i ? C[(i - 1) * G + j] : 0) + (j ? C[i * G + j - 1] : 0) - (i && j ? C[(i - 1) * G + j - 1] : 0);
            T[i * G + j] = mt + (i ? T[(i - 1) * G + j] : 0) + (j ? T[i * G + j - 1] : 0) - (i && j ? T[(i - 1) * G + j - 1] : 0);
            sc += (C[i * G + j] - T[i * G + j]) * (C[i * G + j] - T[i * G + j]);
        }
    return {sd / (G * G), sc / (G * G)};
}

double safe_exp(double l) { return std::isfinite(l) ? std::exp(std::min(l, 50.0)) : 0.0; }
}  // namespace

int main(int argc, char** argv) {
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const std::string cop = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const Fam F = fam_of(cop);
    const Grid g;
    const std::vector<std::string> comps{"xiao", "hermite", "mle"};
    std::vector<std::string> methods;
    for (const auto& c : comps) {
        for (int n : kNeff) methods.push_back(c + "_neff" + std::to_string(n));
        methods.push_back(c + "_bank_own");
        methods.push_back(c + "_bank_log_gauss");
    }
    std::FILE* out = std::fopen(("results/ise" + std::string(cop == "gaussian" ? "" : "_" + cop) + ".csv").c_str(), "w");
    std::fprintf(out, "copula,scenario,seed,window,method,ise_density,ise_cdf\n");
    std::vector<std::string> scs = F.switches ? std::vector<std::string>{"shape"} : std::vector<std::string>{"constant", "drift", "jump"};
    for (const auto& sc : scs)
        for (int s = 0; s < seeds; ++s) {
            std::unique_ptr<Dependence> dep;
            if (sc == "drift") dep = std::make_unique<DriftDependence>(-0.5, 0.8, kN);
            else if (sc == "jump") dep = std::make_unique<JumpDependence>(0.2, 0.8, kChange);
            else dep = std::make_unique<ConstantDependence>(0.6);
            auto c = dgp::make_copula(cop);
            CopulaSource src(*dep, *c, 42 + s);
            std::vector<std::unique_ptr<DependenceEstimator>> own, lg;
            for (const auto& cp : comps) { own.push_back(make_estimator(cp + "_bank_own")); lg.push_back(make_estimator(cp + "_bank_log")); }
            const std::size_t M = methods.size();
            std::vector<double> d_all(M, 0), c_all(M, 0), d_post(M, 0), c_post(M, 0);
            std::size_t n_all = 0, n_post = 0;
            std::vector<double> truth(G * G), est(G * G);
            std::vector<std::vector<double>> arm(5, std::vector<double>(G * G));
            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                if (t >= kBurn && (t - kBurn) % kEvery == 0) {
                    const copula::Family fam = (F.switches && t >= kChange) ? F.after : F.before;
                    const double par = copula::param_from_spearman(fam, dgp::detail::target_spearman(smp.rho));
                    for (std::size_t i = 0; i < G; ++i)
                        for (std::size_t j = 0; j < G; ++j)
                            truth[i * G + j] = safe_exp(copula::log_density(fam, g.u[i], g.u[j], par));
                    const bool post = t >= kChange;
                    for (std::size_t ci = 0; ci < comps.size(); ++ci) {
                        auto* b = dynamic_cast<ForgettingBank*>(own[ci].get());
                        std::vector<double> mix(G * G, 0.0);
                        for (std::size_t k = 0; k < 5; ++k) {
                            const auto* pd = dynamic_cast<const PredictiveDensity*>(&b->model(k));
                            for (std::size_t i = 0; i < G; ++i)
                                for (std::size_t j = 0; j < G; ++j) {
                                    const double v = safe_exp(pd->log_predictive(g.u[i], g.u[j]));
                                    arm[k][i * G + j] = v;
                                    mix[i * G + j] += b->weight(k) * v;
                                }
                        }
                        const double rho = lg[ci]->estimate();
                        const double r = copula::param_from_spearman(copula::Family::Gaussian, rho);
                        for (std::size_t i = 0; i < G; ++i)
                            for (std::size_t j = 0; j < G; ++j)
                                est[i * G + j] = safe_exp(copula::Gaussian::log_density(g.u[i], g.u[j], r));
                        const std::size_t base = ci * 7;
                        auto add = [&](std::size_t m, const std::vector<double>& e) {
                            const auto [di, cdi] = ise(e, truth, g);
                            d_all[m] += di; c_all[m] += cdi;
                            if (post) { d_post[m] += di; c_post[m] += cdi; }
                        };
                        for (std::size_t k = 0; k < 5; ++k) add(base + k, arm[k]);
                        add(base + 5, mix);
                        add(base + 6, est);
                    }
                    ++n_all; if (post) ++n_post;
                }
                for (auto& e : own) e->update(smp.x, smp.y);
                for (auto& e : lg) e->update(smp.x, smp.y);
            }
            for (std::size_t m = 0; m < M; ++m) {
                std::fprintf(out, "%s,%s,%d,whole,%s,%.6g,%.6g\n", cop.c_str(), sc.c_str(), 42 + s, methods[m].c_str(), d_all[m] / n_all, c_all[m] / n_all);
                if (sc != "constant" && sc != "drift")
                    std::fprintf(out, "%s,%s,%d,post,%s,%.6g,%.6g\n", cop.c_str(), sc.c_str(), 42 + s, methods[m].c_str(), d_post[m] / n_post, c_post[m] / n_post);
            }
        }
    std::fclose(out);
}
