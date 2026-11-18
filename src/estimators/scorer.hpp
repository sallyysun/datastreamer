#pragma once

// Scores the device's forecasts against the ground truth of a simulated stream: per-tick squared error, excess log
// loss at the true PIT and H^2 at the checkpoints, then window means. Evaluation only: it needs the truth a deployed
// device never sees, and it keeps every tick (memory grows with the stream).

#include <algorithm>
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
    // Call before dev.update for the observation with its generating truth.
    void observe(const Device& dev, std::size_t t, double true_spearman, double true_pit_x, double true_pit_y,
                 double generating_parameter) {
        if (rows_.empty()) rows_.resize(dev.sets().size() * dev.sets()[0].size());
        const ScoringObservation truth{true_spearman, true_pit_x, true_pit_y, generating_parameter,
                                       copula::log_density(F_, true_pit_x, true_pit_y, generating_parameter), checkpoint(sc_, t)};
        std::vector<double> f;
        std::size_t r = 0;
        for (const auto& s : dev.sets()) {
            s.forecasts(f);
            for (std::size_t i = 0; i < f.size(); ++i, ++r)
                rows_[r].push_back(score_forecast(h2_, truth, f, i, s.mff_weights()));
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
                    const std::size_t end = std::min<std::size_t>(w.b, rs.size());
                    for (std::size_t t = w.a; t < end; ++t) {
                        a += rs[t].squared_error; b += rs[t].excess;
                        if (!std::isnan(rs[t].h2) && (dense || (t - kBurn) % 50 == 0)) { h += rs[t].h2; ++nh; }
                    }
                    const double n = end > w.a ? double(end - w.a) : 0.0;
                    std::fprintf(out, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", fam_.c_str(), sc_.c_str(), s.name().c_str(),
                                 names[i].c_str(), seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                }
            }
            r0 += s.size();
        }
    }

private:
    std::string fam_, sc_;
    copula::Family F_;
    H2Table h2_;
    std::vector<std::vector<ForecastScore>> rows_;
};

}  // namespace thesis
