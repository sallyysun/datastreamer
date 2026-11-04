// One run of the family bank through a shape switch, tick by tick, for the path figure (plot_family_path.py):
// chi_L(0.05) and chi_U(0.05) of the truth, the family bank, the Hermite own-density bank (read off its predictive
// density every 10 ticks) and the corner counter tail_chi_1999; the family bank's summed shape weights, its mean
// memory sum_k w_k M_k and its mean forgetting factor sum_k w_k lambda_k (retention, lambda_k = 1 - 2/(M_k + 1)). Same DGP as probe_tail_bank (copula space, rho_S = 0.582, switch at 5000 of 10000).
// Usage (from experiments/): ./probe_family_path <regime> <seed>  -> results/family_path_<regime>_<seed>.csv

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "tail_common.hpp"

namespace {

constexpr std::size_t T = 10000, TAU = 5000;
constexpr double kQ = 0.05;
const char* kShapeName[] = {"gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90", "clayton270", "gumbel270"};

copula::Shape shape_of(const std::string& f) {
    return f == "t" ? copula::Shape::StudentT : f == "clayton" ? copula::Shape::Clayton
         : f == "gumbel" ? copula::Shape::Gumbel : copula::Shape::Gaussian;
}

// chi_L, chi_U of a density on the unit square (as probe_tail_bank).
template <typename F>
std::pair<double, double> chi_density(F&& f, double q) {
    constexpr int kE = 16, kM = 32;
    std::vector<std::pair<double, double>> vs;
    for (int i = 0; i < kE; ++i) vs.push_back({q * (i + 0.5) / kE, q / kE});
    for (int i = 0; i < kM; ++i) vs.push_back({q + (1 - 2 * q) * (i + 0.5) / kM, (1 - 2 * q) / kM});
    for (int i = 0; i < kE; ++i) vs.push_back({1 - q + q * (i + 0.5) / kE, q / kE});
    double nl = 0, dl = 0, nu = 0, du = 0;
    for (int i = 0; i < kE; ++i) {
        const double ul = q * (i + 0.5) / kE, uh = 1 - q + q * (i + 0.5) / kE;
        for (std::size_t j = 0; j < vs.size(); ++j) {
            const double a = f(ul, vs[j].first) * vs[j].second, b = f(uh, vs[j].first) * vs[j].second;
            dl += a; du += b;
            if (j < std::size_t(kE)) nl += a;
            if (j >= vs.size() - kE) nu += b;
        }
    }
    return {nl / dl, nu / du};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_family_path <regime> <seed>\n"); return 1; }
    const std::string r = argv[1];
    const unsigned seed = static_cast<unsigned>(std::atoi(argv[2]));
    const std::string before = r == "clayton2gumbel" ? "clayton" : "gaussian";
    const std::string after = r == "gauss2t" ? "t" : r == "clayton2gumbel" ? "gumbel" : "clayton";
    tail::Sampler pre(tail::fix(before, 0.6), seed), post(tail::fix(after, 0.6), seed + 100000u);
    const auto tb = copula::exact_tails(shape_of(before), tail::fix(before, 0.6).param, tail::NU, kQ);
    const auto ta = copula::exact_tails(shape_of(after), tail::fix(after, 0.6).param, tail::NU, kQ);

    auto e_fb = make_estimator("family_bank");
    auto* fb = dynamic_cast<FamilyBank*>(e_fb.get());
    auto e_hb = make_estimator("hermite_bank_own");
    auto* hb = dynamic_cast<ForgettingBank*>(e_hb.get());
    auto e_tc = make_estimator("tail_chi_1999");
    const auto* tc = dynamic_cast<const TailEstimator*>(e_tc.get());

    std::FILE* out = std::fopen(("results/family_path_" + r + "_" + std::to_string(seed) + ".csv").c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "t,true_L,true_U,family_L,family_U,hermite_L,hermite_U,counter_L,counter_U,mem,lambda");
    for (const char* n : kShapeName) std::fprintf(out, ",w_%s", n);
    std::fprintf(out, "\n");
    std::pair<double, double> hc{kQ, kQ};
    for (std::size_t t = 0; t < T; ++t) {
        const auto [u, v] = t < TAU ? pre.next() : post.next();
        const auto& tt = t < TAU ? tb : ta;
        if (t % 10 == 0) hc = chi_density([&](double a, double b) { return std::exp(hb->log_density(a, b)); }, kQ);
        const auto ft = fb->tails(kQ);
        const auto ct = tc->tails(kQ);
        std::vector<double> sw(copula::kShapes, 0.0);
        double mem = 0, lam = 0;
        for (std::size_t k = 0; k < fb->size(); ++k) {
            sw[int(fb->arm(k).shape)] += fb->weight(k);
            mem += fb->weight(k) * fb->arm(k).mem;
            lam += fb->weight(k) * (1.0 - 2.0 / (fb->arm(k).mem + 1.0));
        }
        std::fprintf(out, "%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.6f", t, tt.chi_L, tt.chi_U, ft.chi_L, ft.chi_U,
                     hc.first, hc.second, ct.chi_L, ct.chi_U, mem, lam);
        for (double w : sw) std::fprintf(out, ",%.4f", w);
        std::fprintf(out, "\n");
        const double x = special::normal_ppf(u), y = special::normal_ppf(v);
        fb->update(x, y);
        hb->update(x, y);
        e_tc->update(x, y);
    }
    std::fclose(out);
    return 0;
}
