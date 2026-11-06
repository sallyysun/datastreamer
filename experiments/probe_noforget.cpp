// Does forgetting beat no forgetting? Each estimator's log bank and its M = 199 arm against the same
// estimator with no forgetting (cumulative), on the probe_thesis_bank streams (seeds 42-101).
// The cumulative estimators get the bank's inputs (probit of the 500-tick rank PIT), as the arms do;
// raw Xiao on its own frozen cut-points is reported too. See notes/studies/budget_mpl.md §14.
// Usage (from experiments/): ./probe_noforget [seeds=60]  (COPULA=...)  -> results/noforget[_fam].csv
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
std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}
struct PitFed {   // the bank's inputs: probit of a 500-tick rank PIT
    std::unique_ptr<DependenceEstimator> e;
    RankPit px{500}, py{500};
    void update(double x, double y) { e->update(RankPit::probit(px.transform(x)), RankPit::probit(py.transform(y))); }
};
std::unique_ptr<DependenceEstimator> cum_mpl() {
    RecursiveMPL::Config c{};
    c.margin = RecursiveMPL::Margin::ExactNormal;
    c.gain = 1e-12;     // with startup the gain is 1/m: a running mean, i.e. no forgetting
    c.startup = true;
    return std::make_unique<RecursiveMPL>(c, "mpl_cum");
}
}  // namespace

int main(int argc, char** argv) {
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    std::FILE* out = std::fopen(("results/noforget" + std::string(cop == "gaussian" ? "" : "_" + cop) + ".csv").c_str(), "w");
    std::fprintf(out, "copula,scenario,seed,window,method,rmse\n");
    const std::vector<std::string> names{"xiao_raw_cum", "xiao_cum", "xiao_m199", "xiao_bank_log",
                                         "hermite_cum", "hermite_m199", "hermite_bank_log",
                                         "mpl_cum", "mpl_m199", "mpl_bank_log",
                                         "xiao_bankcum_log", "hermite_bankcum_log", "mpl_bankcum_log",
                                         "xiao_bankcum_sq", "mpl_bankcum_sq", "xiao_bank_sq", "mpl_bank_sq"};
    for (const std::string sc : {"constant", "drift", "jump", "oscillate"})
        for (int s = 0; s < seeds; ++s) {
            auto dep = make_dep(sc);
            auto c = dgp::make_copula(cop);
            CopulaSource src(*dep, *c, 42 + s);
            auto xraw = make_estimator("xiao");                // own frozen cut-points, raw inputs
            PitFed xcum{make_estimator("xiao")}, hcum{make_estimator("hermite")}, mcum{cum_mpl()};
            auto xb = make_estimator("xiao_bank_log"), hb = make_estimator("hermite_bank_log"), mb = make_estimator("mle_bank_log");
            auto* xbb = dynamic_cast<ForgettingBank*>(xb.get());
            auto* hbb = dynamic_cast<ForgettingBank*>(hb.get());
            auto* mbb = dynamic_cast<ForgettingBank*>(mb.get());
            auto xc = make_estimator("xiaocum_bank_log"), hc = make_estimator("hermitecum_bank_log"), mc = make_estimator("mlecum_bank_log");
            auto xcs = make_estimator("xiaocum_bank"), mcs = make_estimator("mlecum_bank");
            auto xs = make_estimator("xiao_bank"), ms = make_estimator("mle_bank");
            auto* xcb = dynamic_cast<ForgettingBank*>(xc.get());
            auto* mcb = dynamic_cast<ForgettingBank*>(mc.get());
            double wcum_x = 0, wcum_m = 0, wcum_x_post = 0, wcum_m_post = 0;
            std::vector<double> se_all(names.size(), 0.0), se_post(names.size(), 0.0), se_early(names.size(), 0.0);
            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                // estimates held before the tick; arm 2 of each bank is M = 199
                const double r[] = {xraw->estimate(), xcum.e->estimate(), xbb->model(2).estimate(), xb->estimate(),
                                    hcum.e->estimate(), hbb->model(2).estimate(), hb->estimate(),
                                    mcum.e->estimate(), mbb->model(2).estimate(), mb->estimate(),
                                    xc->estimate(), hc->estimate(), mc->estimate(),
                                    xcs->estimate(), mcs->estimate(), xs->estimate(), ms->estimate()};
                if (t >= kBurn)
                    for (std::size_t i = 0; i < names.size(); ++i) {
                        const double e2 = (r[i] - smp.rho_s) * (r[i] - smp.rho_s);
                        se_all[i] += e2;
                        if (t >= kChange) se_post[i] += e2;
                        if (t < 2000) se_early[i] += e2;
                    }
                xraw->update(smp.x, smp.y); xcum.update(smp.x, smp.y); hcum.update(smp.x, smp.y); mcum.update(smp.x, smp.y);
                xb->update(smp.x, smp.y); hb->update(smp.x, smp.y); mb->update(smp.x, smp.y);
                xc->update(smp.x, smp.y); hc->update(smp.x, smp.y); mc->update(smp.x, smp.y);
                xcs->update(smp.x, smp.y); mcs->update(smp.x, smp.y); xs->update(smp.x, smp.y); ms->update(smp.x, smp.y);
                if (t >= kBurn) {   // weight on the cumulative arm (index 5)
                    wcum_x += xcb->weight(5); wcum_m += mcb->weight(5);
                    if (t >= kChange) { wcum_x_post += xcb->weight(5); wcum_m_post += mcb->weight(5); }
                }
            }
            std::fprintf(out, "%s,%s,%d,whole,w_cum_xiao,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, wcum_x / (kN - kBurn));
            std::fprintf(out, "%s,%s,%d,whole,w_cum_mpl,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, wcum_m / (kN - kBurn));
            std::fprintf(out, "%s,%s,%d,post,w_cum_xiao,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, wcum_x_post / (kN - kChange));
            std::fprintf(out, "%s,%s,%d,post,w_cum_mpl,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, wcum_m_post / (kN - kChange));
            for (std::size_t i = 0; i < names.size(); ++i) {
                std::fprintf(out, "%s,%s,%d,whole,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, names[i].c_str(), std::sqrt(se_all[i] / (kN - kBurn)));
                std::fprintf(out, "%s,%s,%d,early,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, names[i].c_str(), std::sqrt(se_early[i] / (2000 - kBurn)));
                if (sc == "jump") std::fprintf(out, "%s,%s,%d,post,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, names[i].c_str(), std::sqrt(se_post[i] / (kN - kChange)));
            }
        }
    std::fclose(out);
}
