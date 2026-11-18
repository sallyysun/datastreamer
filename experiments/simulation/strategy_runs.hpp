#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../src/copula_family.hpp"
#include "../../src/estimators/device.hpp"
#include "csv_list.hpp"

namespace thesis {
struct H2Table;
}

namespace strategy_experiments {

struct Run {
    int seeds;
    std::string cop, sfx;
    const char* tag;
    thesis::H2Table& h2;
};

inline std::FILE* open_result(const std::string& name) { return std::fopen(("results/" + name + ".csv").c_str(), "w"); }

inline int missing_results_folder() {
    std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n");
    return 1;
}

struct SeedMeanSd { double mean, sd; };

// Mean and sample standard deviation across seeds from running sums.
inline SeedMeanSd seed_mean_sd(double sum, double sum_sq, int seeds) {
    const double mean = sum / seeds;
    const double var = std::max(0.0, sum_sq / seeds - mean * mean) * seeds / (seeds - 1.0);
    return {mean, std::sqrt(var)};
}

using thesis::FixedMemoryArms;

int run_validate(const Run& run);
int run_family_switching_study(const Run& run);
int run_family_score_comparison(const Run& run);
int run_regretfam(const Run& run);
int run_regret(const Run& run);

}  // namespace strategy_experiments
