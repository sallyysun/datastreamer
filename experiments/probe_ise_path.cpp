// One seed, per estimator, of three banks on the same stream: own-density loss as tuned (*_bank_own,
// eta 0.25, delta 3e-3), MoM log loss (*_bank_log, eta 1, delta 3e-4; density = Gaussian at the bank rho),
// and own-density loss at the log bank's eta/delta. Per tick weights and rho; density ISE (normal-score
// scale) every 20 ticks. Illustration for
// budget_mpl.md §18 (multi-seed numbers: probe_ise). Usage (from experiments/):
//   ./probe_ise_path [seed=42]  (COPULA=gaussian SCENARIO=jump | COPULA=clayton2gumbel SCENARIO=shape)
//   -> results/ise_path_<copula>_<scenario>.csv   comp,bank_type,t,truth,bank,w0..w4,ise
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

int main(int argc, char** argv) {
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const unsigned seed = argc > 1 ? std::atoi(argv[1]) : 42;
    const std::string cop = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const std::string sc = std::getenv("SCENARIO") ? std::getenv("SCENARIO") : "jump";
    using F = copula::Family;
    F before = F::Gaussian, after = F::Gaussian;
    if (cop == "clayton2gumbel") { before = F::Clayton; after = F::Gumbel; }
    else if (cop == "gauss2t") { before = F::Gaussian; after = F::StudentT; }
    else if (cop == "clayton") before = after = F::Clayton;
    const bool sw = cop == "clayton2gumbel" || cop == "gauss2t";
    constexpr std::size_t G = 32;
    std::vector<double> u(G), w(G);
    for (std::size_t i = 0; i < G; ++i) {
        u[i] = (i + 0.5) / G;
        const double z = RankPit::probit(u[i]);
        w[i] = std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI);
    }
    std::FILE* out = std::fopen(("results/ise_path_" + cop + "_" + sc + ".csv").c_str(), "w");
    std::fprintf(out, "comp,bank_type,t,truth,bank,w0,w1,w2,w3,w4,ise\n");
    for (const std::string comp : {"xiao", "hermite", "mle"}) {
        std::unique_ptr<Dependence> dep;
        if (sc == "jump") dep = std::make_unique<JumpDependence>(0.2, 0.8, 2500);
        else if (sc == "drift") dep = std::make_unique<DriftDependence>(-0.5, 0.8, 5000);
        else dep = std::make_unique<ConstantDependence>(0.6);
        auto c = dgp::make_copula(cop);
        CopulaSource src(*dep, *c, seed);
        // three banks on the same stream: own loss as tuned, MoM log loss, own loss at the log bank's eta/delta
        auto own = make_estimator(comp + "_bank_own");
        auto lg = make_estimator(comp + "_bank_log");
        auto* ob = dynamic_cast<ForgettingBank*>(own.get());
        // own loss at eta 1 / delta 3e-4: registry has no such key, so construct one from a log bank's arm recipe
        std::unique_ptr<ForgettingBank> o1;
        {
            std::vector<std::unique_ptr<DependenceEstimator>> m;
            if (comp == "xiao") {
                for (double lam : {0.95, 0.98, 0.99, 0.995, 0.999})
                    m.push_back(std::make_unique<XiaoSpearmanEstimator>(XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = lam, .calibration = 100}, "m"));
            } else if (comp == "hermite") {
                for (double lam : {0.05, 0.02, 0.01, 0.005, 0.001})
                    m.push_back(std::make_unique<HermiteEstimator>(HermiteEstimator::Config{.order = 10, .lambda = lam, .burn_in = 50}, "m"));
            } else {
                for (double gn : {0.05, 0.02, 0.01, 0.005, 0.001}) {
                    RecursiveMPL::Config cfg{}; cfg.margin = RecursiveMPL::Margin::ExactNormal; cfg.gain = gn; cfg.startup = true;
                    m.push_back(std::make_unique<RecursiveMPL>(cfg, "m"));
                }
            }
            o1 = std::make_unique<ForgettingBank>(std::move(m), ForgettingBank::Config{.loss = ForgettingBank::Loss::Own, .eta = 1.0, .delta = 3e-4}, "o1");
        }
        auto* lb = dynamic_cast<ForgettingBank*>(lg.get());
        struct B { const char* name; ForgettingBank* b; bool gauss; };
        const B banks[3] = {{"own", ob, false}, {"log", lb, true}, {"own_eta1", o1.get(), false}};
        for (std::size_t t = 0; t < 5000; ++t) {
            const auto smp = src.next();
            const bool chk = t >= 200 && t % 20 == 0;
            const F fam = (sw && t >= 2500) ? after : before;
            const double par = copula::param_from_spearman(fam, dgp::detail::target_spearman(smp.rho));
            for (const auto& bb : banks) {
                double ise = -1;
                if (chk) {
                    double s = 0;
                    const double r = copula::param_from_spearman(F::Gaussian, bb.b->estimate());
                    for (std::size_t i = 0; i < G; ++i)
                        for (std::size_t j = 0; j < G; ++j) {
                            double mix = 0;
                            if (bb.gauss) mix = std::exp(copula::Gaussian::log_density(u[i], u[j], r));
                            else
                                for (std::size_t k = 0; k < 5; ++k) {
                                    const double l = dynamic_cast<const PredictiveDensity*>(&bb.b->model(k))->log_predictive(u[i], u[j]);
                                    mix += bb.b->weight(k) * (std::isfinite(l) ? std::exp(std::min(l, 50.0)) : 0.0);
                                }
                            const double tr = std::exp(copula::log_density(fam, u[i], u[j], par));
                            s += (mix - tr) * (mix - tr) * w[i] * w[j];
                        }
                    ise = s / (G * G);
                }
                std::fprintf(out, "%s,%s,%zu,%.5f,%.5f", comp.c_str(), bb.name, t, smp.rho_s, bb.b->estimate());
                for (std::size_t k = 0; k < 5; ++k) std::fprintf(out, ",%.5g", bb.b->weight(k));
                std::fprintf(out, ",%.6g\n", ise);
            }
            for (const auto& bb : banks) bb.b->update(smp.x, smp.y);
        }
    }
    std::fclose(out);
}
