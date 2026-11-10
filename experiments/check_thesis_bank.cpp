// Checks that the thesis-facing code does what Chapters 3-5 say:
//   1. ForgettingBank reproduces a hand-written Algorithm 1 (Hedge then fixed share, per tick).
//   2. eta = delta = 0 keeps the weights uniform (the averaging control).
//   3. expected_info_phi matches Monte Carlo E[(d log c / d phi)^2] (section 3.5.4).
//   4. Every registered key constructs.
// Usage (from experiments/): make check_thesis_bank && ./check_thesis_bank

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"

namespace {

struct Const : DependenceEstimator {
    double r;
    explicit Const(double r) : r(r) {}
    void update(double, double) override {}
    double estimate() const override { return r; }
    void reset() override {}
    std::size_t store_bytes() const override { return sizeof(*this); }
    const char* name() const override { return "const"; }
};

int failures = 0;
void expect(bool ok, const char* what, double got, double want) {
    std::printf("%-58s %s  (got %.6g, want %.6g)\n", what, ok ? "ok  " : "FAIL", got, want);
    if (!ok) ++failures;
}

void check_algorithm1(ForgettingBank::Loss loss) {
    const std::vector<double> rhos{-0.2, 0.1, 0.5, 0.8};
    std::vector<std::unique_ptr<DependenceEstimator>> m;
    for (double r : rhos) m.push_back(std::make_unique<Const>(r));
    const ForgettingBank::Config cfg{.loss = loss, .eta = 0.7, .delta = 0.01, .marginal_neff = 200.0, .warm_up = 30};
    ForgettingBank bank(std::move(m), cfg, "b");

    EwMarginal px{200.0}, py{200.0};
    std::vector<double> w(rhos.size(), 0.25);
    std::mt19937 rng(3);
    std::normal_distribution<double> N;
    double maxdiff = 0.0;
    for (int t = 0; t < 3000; ++t) {
        const double z1 = N(rng), x = z1, y = 0.6 * z1 + 0.8 * N(rng);
        const bool scorable = px.size() >= cfg.warm_up;
        const double u = px.transform(x), v = py.transform(y);
        if (scorable) {
            double tot = 0.0;
            std::vector<double> wt(w.size());
            for (std::size_t k = 0; k < w.size(); ++k) {
                double l;
                if (loss == ForgettingBank::Loss::Squared) {
                    const double p = 12 * (u - .5) * (v - .5);
                    l = (p - rhos[k]) * (p - rhos[k]);
                } else {
                    l = -copula::Gaussian::log_density(u, v, 2 * std::sin(std::numbers::pi * rhos[k] / 6));
                }
                wt[k] = w[k] * std::exp(-cfg.eta * l);
                tot += wt[k];
            }
            for (std::size_t k = 0; k < w.size(); ++k) w[k] = (1 - cfg.delta) * wt[k] / tot + cfg.delta / w.size();
        }
        bank.update(x, y);
        for (std::size_t k = 0; k < w.size(); ++k) maxdiff = std::max(maxdiff, std::abs(bank.weight(k) - w[k]));
    }
    expect(maxdiff < 1e-12, loss == ForgettingBank::Loss::Squared
                                ? "1. Algorithm 1, squared loss: max |w_bank - w_hand|"
                                : "1. Algorithm 1, log loss: max |w_bank - w_hand|", maxdiff, 0.0);
}

void check_uniform() {
    auto b = make_estimator("xiao_bank_uniform");
    auto* bank = dynamic_cast<ForgettingBank*>(b.get());
    std::mt19937 rng(5);
    std::normal_distribution<double> N;
    for (int t = 0; t < 2000; ++t) { const double z = N(rng); bank->update(z, 0.5 * z + N(rng)); }
    double dev = 0.0;
    for (std::size_t k = 0; k < bank->size(); ++k) dev = std::max(dev, std::abs(bank->weight(k) - 0.2));
    expect(dev < 1e-15, "2. eta = delta = 0: max |w - 1/K|", dev, 0.0);
}

void check_info(const char* fam, copula::Family F, double d) {
    auto cop = dgp::make_copula(fam);
    const double par = copula::param_from_spearman(F, dgp::detail::target_spearman(d), 4.0);
    std::mt19937 rng(11);
    double s2 = 0.0;
    const int n = 400000;
    for (int i = 0; i < n; ++i) {
        const auto dr = cop->sample(d, rng);
        const double u = std::clamp(special::normal_cdf(dr.z1), 1e-12, 1 - 1e-12);
        const double v = std::clamp(special::normal_cdf(dr.z2), 1e-12, 1 - 1e-12);
        double s = 0.0;
        switch (F) {
            case copula::Family::Clayton:  s = copula::Clayton::score(u, v, par) * par; break;
            case copula::Family::Gumbel:   s = copula::Gumbel::score(u, v, par) * (par - 1); break;
            case copula::Family::StudentT: s = copula::StudentT::score(u, v, par, 4.0) * (1 - par * par); break;
            default:                       s = copula::Gaussian::score(u, v, par) * (1 - par * par); break;
        }
        if (std::isfinite(s)) s2 += s * s;
    }
    const double mc = s2 / n, tab = copula::expected_info_phi(F, par, 4.0);
    char what[96];
    std::snprintf(what, sizeof what, "3. I(phi) %-8s param %.3f: table vs Monte Carlo", fam, par);
    expect(std::abs(tab / mc - 1) < 0.05, what, tab, mc);
}

}  // namespace

int main() {
    check_algorithm1(ForgettingBank::Loss::Squared);
    check_algorithm1(ForgettingBank::Loss::Log);
    check_uniform();
    for (double d : {0.3, 0.6, 0.85}) {
        check_info("gaussian", copula::Family::Gaussian, d);
        check_info("clayton", copula::Family::Clayton, d);
        check_info("gumbel", copula::Family::Gumbel, d);
        check_info("t", copula::Family::StudentT, d);
    }
    expect(check_registry(), "4. every registered key constructs", 1, 1);
    std::printf("%s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
