#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../src/copula_family.hpp"
#include "../../src/estimators/device.hpp"

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

inline constexpr const char* kEst[] = {"xiao", "hermite", "rmpl"};

using thesis::FixedMemoryArms;

int run_famswitch(const Run& run);
int run_scoresfam(const Run& run);
int run_regretfam(const Run& run);
int run_regret(const Run& run);

using thesis::FamilyHellinger;
using thesis::FamilyInverse;

}  // namespace strategy_experiments
