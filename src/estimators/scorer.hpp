#pragma once

// Scores the device's forecasts against the ground truth of a simulated stream: per-tick squared error, excess log
// loss at the true PIT and H^2 at the checkpoints, then window means. Evaluation only: it needs the truth a deployed
// device never sees, and it keeps every tick (memory grows with the stream).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "device.hpp"
#include "evaluation.hpp"

namespace thesis {

// Per-tick squared error, excess log loss at the true PIT and H^2 at the checkpoints, then window means.
class Scorer {
public:
    Scorer(const std::string& family, const std::string& scenario)
        : fam_(family), sc_(scenario), F_(family_of(family)), h2_{.F = family_of(family)} {}
    // Call before dev.update for the observation with truth (rho_s, pit_x, pit_y, par).
    void observe(const Device& dev, std::size_t t, double rho_s, double pu, double pv, double par) {
        if (rows_.empty()) rows_.resize(dev.sets().size() * dev.sets()[0].size());
        const double lc = copula::log_density(F_, pu, pv, par);
        const bool cp = checkpoint(sc_, t);
        std::vector<double> f;
        std::size_t r = 0;
        for (const auto& s : dev.sets()) {
            s.forecasts(f);
            for (std::size_t i = 0; i < f.size(); ++i, ++r) {
                const double d = f[i] - rho_s;
                double ex, hh = NAN;
                if (i == kK) {
                    ex = lc - H2Table::mix_logc(f.data(), s.mff_weights().data(), kK, pu, pv);
                    if (cp) hh = h2_.mix(par, f.data(), s.mff_weights().data(), kK);
                } else {
                    ex = lc - gauss_logc(f[i], pu, pv);
                    if (cp) hh = h2_(par, f[i]);
                }
                rows_[r].push_back({d * d, ex, hh});
            }
        }
    }
    // The runner's strategies_seeds rows: copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000
    void write(std::FILE* out, const Device& dev, unsigned seed) const {
        const auto& names = strategy_names();
        std::size_t r0 = 0;
        for (const auto& s : dev.sets()) {
            for (const auto& w : windows(sc_)) {
                const bool dense = std::string(w.name) == "rec_0_100";
                for (std::size_t i = 0; i < s.size(); ++i) {
                    const auto& rs = rows_[r0 + i];
                    double a = 0, b = 0, h = 0; int nh = 0;
                    for (std::size_t t = w.a; t < w.b && t < rs.size(); ++t) {
                        a += rs[t].se; b += rs[t].ex;
                        if (!std::isnan(rs[t].h2) && (dense || (t - kBurn) % 50 == 0)) { h += rs[t].h2; ++nh; }
                    }
                    const double n = double(w.b - w.a);
                    std::fprintf(out, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", fam_.c_str(), sc_.c_str(), s.name().c_str(),
                                 names[i].c_str(), seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                }
            }
            r0 += s.size();
        }
    }

private:
    struct Tick { double se, ex, h2; };
    std::string fam_, sc_;
    copula::Family F_;
    H2Table h2_;
    std::vector<std::vector<Tick>> rows_;
};

}  // namespace thesis
