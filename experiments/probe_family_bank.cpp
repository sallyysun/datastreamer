// The family bank (FamilyBank: copula shapes x memories, own copula log loss, eta = 1, delta*) against
// the Chapter 6 forgetting banks, all on the same streams. Per window: rho_S RMSE, excess log loss at
// the true PIT of each bank's predictive density (log c_true - log c_bank), and for the family bank and
// an EW ratio estimator chi_hat(q) = J/K at fixed memories, the RMSE of chi_L(q), chi_U(q) against the
// true finite-q value (q = 0.05). Shape switches (COPULA=gauss2t|clayton2gumbel|gauss2clayton) hold
// rho_S at the d = 0.6 value and switch family at tick 2500.
// Note: notes/studies/thesis_bank_family.md.
//
// Usage (from experiments/):  ./probe_family_bank <first_seed> <seeds>
//   (COPULA=gaussian|clayton|gumbel|t|gauss2t|clayton2gumbel|gauss2clayton)
//   -> results/family_bank_<first_seed>[_<family>].csv          seed x scenario x window x method x metric
//      results/family_bank_path_<first_seed>[_<family>].csv     seed-mean shape weights, memory, tails per tick
//      results/family_bank_switch_<first_seed>[_<family>].csv   shape switches: ticks for the new shape to
//                                                               reach 50% / 90% of the weight; on the constant
//                                                               path (no switch), the same for every other
//                                                               shape from tick 2500 (false switches)

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
#include "scenarios.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kWarm = 30;
constexpr double kQ = 0.05, kNu = 4.0;
const std::vector<std::string> kBanks{"xiao_bank_log", "hermite_bank_log", "mle_bank_log",
                                      "xiao_bank_own", "hermite_bank_own", "mle_bank_own"};
const std::vector<int> kRatioMem{99, 399, 1999};
const char* kShapeName[] = {"gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90", "clayton270", "gumbel270"};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant" || s == "shape") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return scenario_drift(kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return scenario_oscillate();
}

copula::Shape shape_of(const std::string& f) {
    if (f == "clayton") return copula::Shape::Clayton;
    if (f == "gumbel") return copula::Shape::Gumbel;
    if (f == "t") return copula::Shape::StudentT;
    return copula::Shape::Gaussian;
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump" || s == "shape") {
        w.push_back({"pre", kBurn, kChange});
        w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500});
        w.push_back({"rec_500_1500", kChange + 500, kChange + 1500});
        w.push_back({"post", kChange, kN});
    }
    return w;
}

// EW ratio estimator of chi(q): joint corner events over marginal corner events, both discounted.
struct Ratio {
    double lam, jl = 0, kl = 0, ju = 0, ku = 0;
    double chi_L() const { return kl > 0 ? jl / kl : 0.0; }
    double chi_U() const { return ku > 0 ? ju / ku : 0.0; }
    void update(double u, double v) {
        jl = lam * jl + (u <= kQ && v <= kQ);
        kl = lam * kl + (u <= kQ);
        ju = lam * ju + (u > 1 - kQ && v > 1 - kQ);
        ku = lam * ku + (u > 1 - kQ);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_family_bank <first_seed> <seeds>\n"); return 1; }
    const unsigned first = static_cast<unsigned>(std::atoi(argv[1]));
    const int seeds = std::atoi(argv[2]);
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const bool shape = cop_name == "gauss2t" || cop_name == "clayton2gumbel" || cop_name == "gauss2clayton";
    const copula::Shape S0 = cop_name == "gauss2t" || cop_name == "gauss2clayton" ? copula::Shape::Gaussian
                           : cop_name == "clayton2gumbel" ? copula::Shape::Clayton : shape_of(cop_name);
    const copula::Shape S1 = cop_name == "gauss2t" ? copula::Shape::StudentT
                           : cop_name == "clayton2gumbel" ? copula::Shape::Gumbel
                           : cop_name == "gauss2clayton" ? copula::Shape::Clayton : S0;
    const std::vector<std::string> scenarios = shape ? std::vector<std::string>{"shape"}
        : std::vector<std::string>{"constant", "drift", "jump", "oscillate"};
    const std::string sfx = (cop_name == "gaussian" ? "" : "_" + cop_name);
    const std::string tag = std::to_string(first) + (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string()) + sfx + ".csv";

    std::FILE* out = std::fopen(("results/family_bank_" + tag).c_str(), "w");
    std::FILE* pout = std::fopen(("results/family_bank_path_" + tag).c_str(), "w");
    std::FILE* sout = std::fopen(("results/family_bank_switch_" + tag).c_str(), "w");
    if (!out || !pout || !sout) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,window,method,metric,value\n");
    std::fprintf(pout, "copula,scenario,t,series,value\n");
    std::fprintf(sout, "copula,scenario,seed,new_shape,t50,t90,pre_weight_new\n");

    for (const std::string& sc : scenarios) {
        // Seed means per tick: 6 shape weights, memory, truth / bank / ratio_399 chi_L and chi_U, rho.
        std::vector<std::string> series;
        for (const char* n : kShapeName) series.push_back(std::string("w_") + n);
        for (const char* n : {"mem", "rho_true", "rho_family", "chiL_true", "chiL_family", "chiL_ratio399",
                              "chiU_true", "chiU_family", "chiU_ratio399", "lamL_family", "lamU_family"})
            series.push_back(n);
        std::vector<std::vector<double>> path(series.size(), std::vector<double>(kN, 0.0));
        for (int s = 0; s < seeds; ++s) {
            auto dep = make_dep(sc);
            auto cop = dgp::make_copula(cop_name);
            CopulaSource src(*dep, *cop, first + s);
            auto e_fb = make_estimator("family_bank");
            auto* fb = dynamic_cast<FamilyBank*>(e_fb.get());
            const std::size_t K = fb->size();
            std::vector<std::unique_ptr<DependenceEstimator>> banks;
            for (const auto& k : kBanks) banks.push_back(make_estimator(k));
            EwMarginal px{default_marginal_neff()}, py{default_marginal_neff()};
            std::vector<Ratio> ratios;
            for (int m : kRatioMem) ratios.push_back({1.0 - 2.0 / (m + 1.0)});

            // methods: family_bank, the banks, fixed family arms; metrics per tick
            std::vector<std::string> rho_m{"family_bank"};
            for (const auto& k : kBanks) rho_m.push_back(k);
            for (std::size_t k = 0; k < K; ++k)
                rho_m.push_back(std::string("arm_") + kShapeName[int(fb->arm(k).shape)] + "_" + std::to_string(fb->arm(k).mem));
            std::vector<std::string> chi_m{"family_bank"};
            for (int m : kRatioMem) chi_m.push_back("ratio_" + std::to_string(m));
            const std::size_t nb = 1 + kBanks.size();
            std::vector<std::vector<float>> se(rho_m.size(), std::vector<float>(kN)), ex(nb, std::vector<float>(kN)),
                cl(chi_m.size(), std::vector<float>(kN)), cu(chi_m.size(), std::vector<float>(kN));
            std::vector<long> t50(copula::kShapes, -1), t90(copula::kShapes, -1);
            std::vector<double> pre_w(copula::kShapes, 0.0);
            const bool null_path = !shape && sc == "constant";

            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                const copula::Shape St = t < kChange ? S0 : S1;
                const copula::Family Ft = copula::base_family(St);
                const double par = copula::param_from_spearman(Ft, dgp::detail::target_spearman(smp.rho), kNu);
                const auto tt = copula::exact_tails(St, par, kNu, kQ);
                const double u0 = special::normal_cdf(smp.x), v0 = special::normal_cdf(smp.y);
                const double lc = copula::log_density(Ft, u0, v0, par, kNu);

                // Predictions held before this tick.
                se[0][t] = float(std::pow(fb->estimate() - smp.rho_s, 2));
                ex[0][t] = float(lc - fb->log_density(u0, v0));
                for (std::size_t b = 0; b < kBanks.size(); ++b) {
                    auto* fbk = dynamic_cast<ForgettingBank*>(banks[b].get());
                    se[1 + b][t] = float(std::pow(fbk->estimate() - smp.rho_s, 2));
                    ex[1 + b][t] = float(lc - fbk->log_density(u0, v0));
                }
                for (std::size_t k = 0; k < K; ++k) se[nb + k][t] = float(std::pow(fb->arm_rho(k) - smp.rho_s, 2));
                const auto ft = fb->tails(kQ);
                cl[0][t] = float(std::pow(ft.chi_L - tt.chi_L, 2));
                cu[0][t] = float(std::pow(ft.chi_U - tt.chi_U, 2));
                for (std::size_t r = 0; r < ratios.size(); ++r) {
                    cl[1 + r][t] = float(std::pow(ratios[r].chi_L() - tt.chi_L, 2));
                    cu[1 + r][t] = float(std::pow(ratios[r].chi_U() - tt.chi_U, 2));
                }

                std::vector<double> sw(copula::kShapes, 0.0);
                double mem = 0;
                for (std::size_t k = 0; k < K; ++k) {
                    sw[int(fb->arm(k).shape)] += fb->weight(k);
                    mem += fb->weight(k) * fb->arm(k).mem;
                }
                std::vector<double> vals(sw);
                for (double x : {mem, smp.rho_s, fb->estimate(), tt.chi_L, ft.chi_L, ratios[1].chi_L(), tt.chi_U,
                                 ft.chi_U, ratios[1].chi_U(), ft.lambda_L, ft.lambda_U})
                    vals.push_back(x);
                for (std::size_t i = 0; i < series.size(); ++i) path[i][t] += vals[i] / seeds;
                for (int j = 0; j < copula::kShapes; ++j) {
                    if (t == kChange - 1) pre_w[j] = sw[j];
                    if (t >= kChange && t50[j] < 0 && sw[j] >= 0.5) t50[j] = long(t - kChange);
                    if (t >= kChange && t90[j] < 0 && sw[j] >= 0.9) t90[j] = long(t - kChange);
                }

                fb->update(smp.x, smp.y);
                for (auto& b : banks) b->update(smp.x, smp.y);
                const bool scorable = px.size() >= kWarm;
                const double u = px.transform(smp.x), v = py.transform(smp.y);
                if (scorable) for (auto& r : ratios) r.update(u, v);
            }
            for (int j = 0; j < copula::kShapes; ++j)
                if ((shape && j == int(S1)) || (null_path && j != int(S0)))
                    std::fprintf(sout, "%s,%s,%u,%s,%ld,%ld,%.4f\n", cop_name.c_str(), sc.c_str(), first + s,
                                 kShapeName[j], t50[j], t90[j], pre_w[j]);
            for (const auto& w : windows(sc)) {
                const double n = double(w.b - w.a);
                auto emit = [&](const std::string& m, const char* metric, const std::vector<float>& x, double scale, bool root) {
                    double a = 0;
                    for (std::size_t t = w.a; t < w.b; ++t) a += x[t];
                    a /= n;
                    std::fprintf(out, "%s,%s,%u,%s,%s,%s,%.6f\n", cop_name.c_str(), sc.c_str(), first + s, w.name,
                                 m.c_str(), metric, scale * (root ? std::sqrt(a) : a));
                };
                for (std::size_t i = 0; i < rho_m.size(); ++i) emit(rho_m[i], "rmse", se[i], 1.0, true);
                for (std::size_t i = 0; i < nb; ++i) emit(rho_m[i], "excess_x1000", ex[i], 1000.0, false);
                for (std::size_t i = 0; i < chi_m.size(); ++i) {
                    emit(chi_m[i], "chiL_rmse", cl[i], 1.0, true);
                    emit(chi_m[i], "chiU_rmse", cu[i], 1.0, true);
                }
            }
        }
        for (std::size_t t = 0; t < kN; t += 10)
            for (std::size_t i = 0; i < series.size(); ++i)
                std::fprintf(pout, "%s,%s,%zu,%s,%.5f\n", cop_name.c_str(), sc.c_str(), t, series[i].c_str(), path[i][t]);
        std::fprintf(stderr, "%s %s done\n", cop_name.c_str(), sc.c_str());
    }
    std::fclose(out);
    std::fclose(pout);
    std::fclose(sout);
    return 0;
}
