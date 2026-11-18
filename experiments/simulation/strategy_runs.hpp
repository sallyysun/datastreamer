#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../src/copula_family.hpp"
#include "../../src/estimators/device.hpp"
#include "study_stream.hpp"

namespace thesis {
struct H2Table;
struct Win;
}

namespace strategy_experiments {

struct Run {
    int seeds;
    std::string cop, sfx;
    const char* tag;
    thesis::H2Table& h2;
};

struct Stream : StudySeries { std::vector<double> par; };

inline constexpr const char* kEst[] = {"xiao", "hermite", "rmpl"};
Stream make_stream(const std::string& family, const std::string& scenario, unsigned seed);

using thesis::FixedMemoryArms;

struct WindowScores {
    double rmse, excess_x1000, h2_x1000;
};

struct StrategyTrace {
    std::vector<double> forecast, squared_error, excess, h2;
};

struct EstimatorTrace {
    std::vector<StrategyTrace> strategies;
    double mixture_log_loss = 0;
};

struct SeedResults {
    unsigned seed;
    std::vector<double> rho;
    std::vector<EstimatorTrace> estimators;
};

// One scenario and seed: replay the series into each estimator's strategies, scoring each held forecast first.
class Simulation {
public:
    Simulation(const Run& run, std::string scenario, unsigned seed, const std::vector<double>& rates, bool mixture);
    SeedResults run() const;

private:
    const Run& run_;
    std::string scenario_;
    unsigned seed_;
    const std::vector<double>& rates_;
    bool mixture_;
};

WindowScores summarize_window(const thesis::Win& window, const std::vector<double>& squared_error,
                              const std::vector<double>& excess, const std::vector<double>& hellinger);

int run_famswitch(const Run& run);
int run_scoresfam(const Run& run);
int run_regretfam(const Run& run);
int run_regret(const Run& run);

// rho_S -> family parameter, tabulated on 40001 points between the family's parameter bounds (exact for the Gaussian):
// how each RMPL arm's rho_S becomes a working-family density in 6.4-6.5.
struct FamilyInverse {
    copula::Family F;
    double lo = 0, hi = 0;
    std::vector<double> th;
    explicit FamilyInverse(copula::Family f) : F(f) {
        if (F == copula::Family::Gaussian) return;
        const double plo = F == copula::Family::Clayton ? copula::Clayton::kMinParam : F == copula::Family::Gumbel ? copula::Gumbel::kMinParam : copula::StudentT::kMinParam;
        const double phi = F == copula::Family::Clayton ? copula::Clayton::kMaxParam : F == copula::Family::Gumbel ? copula::Gumbel::kMaxParam : copula::StudentT::kMaxParam;
        lo = copula::spearman_from_param(F, plo);
        hi = copula::spearman_from_param(F, phi);
        for (int i = 0; i < 40001; ++i) th.push_back(copula::param_from_spearman(F, lo + (hi - lo) * i / 40000.0));
    }
    bool clamped(double rho) const { return F != copula::Family::Gaussian && (rho <= lo || rho >= hi); }
    double operator()(double rho) const {
        if (F == copula::Family::Gaussian) return copula::param_from_spearman(F, rho);
        const double x = (std::clamp(rho, lo, hi) - lo) / (hi - lo) * (th.size() - 1.0);
        const std::size_t i = std::min(th.size() - 2, std::size_t(x));
        return th[i] + (x - i) * (th[i + 1] - th[i]);
    }
};

}  // namespace strategy_experiments
