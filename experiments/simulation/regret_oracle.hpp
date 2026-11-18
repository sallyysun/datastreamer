#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline constexpr int kSwitchBudgets = 3;

struct RegretOracle {
    std::vector<std::array<double, kSwitchBudgets>> prefix_loss;
    double fixed_loss, one_switch_loss;
    std::size_t fixed_arm, switch_index, before_arm, after_arm;
};

inline RegretOracle regret_oracle(const std::vector<std::vector<double>>& losses) {
    if (losses.empty() || losses.front().empty()) {
        throw std::invalid_argument("regret oracle needs losses for at least one tick and arm");
    }
    const std::size_t ticks = losses.size(), arms = losses.front().size();
    constexpr double infinity = std::numeric_limits<double>::infinity();
    RegretOracle oracle{};
    oracle.prefix_loss.resize(ticks);
    oracle.one_switch_loss = infinity;

    // State is the minimum loss for each final arm and exact switch count.
    std::vector<double> previous(kSwitchBudgets * arms, infinity), current(previous.size());
    for (std::size_t t = 0; t < ticks; ++t) {
        double best = infinity;
        for (int switches = 0; switches < kSwitchBudgets; ++switches) {
            double change = infinity;
            if (switches > 0) {
                for (std::size_t arm = 0; arm < arms; ++arm) {
                    change = std::min(change, previous[(switches - 1) * arms + arm]);
                }
            }
            for (std::size_t arm = 0; arm < arms; ++arm) {
                const auto index = switches * arms + arm;
                const double start = t == 0 && switches == 0 ? 0.0 : previous[index];
                current[index] = losses[t][arm] + std::min(start, change);
                best = std::min(best, current[index]);
            }
            oracle.prefix_loss[t][switches] = best;
        }
        previous.swap(current);
    }

    // Prefix sums identify the static arm and the first best one-switch path.
    std::vector<std::vector<double>> cumulative(arms, std::vector<double>(ticks + 1, 0.0));
    for (std::size_t arm = 0; arm < arms; ++arm) {
        for (std::size_t t = 0; t < ticks; ++t) {
            cumulative[arm][t + 1] = cumulative[arm][t] + losses[t][arm];
        }
        if (cumulative[arm][ticks] < cumulative[oracle.fixed_arm][ticks]) oracle.fixed_arm = arm;
    }
    oracle.fixed_loss = cumulative[oracle.fixed_arm][ticks];

    for (std::size_t t = 0; t <= ticks; ++t) {
        for (std::size_t before = 0; before < arms; ++before) {
            for (std::size_t after = 0; after < arms; ++after) {
                const double loss = cumulative[before][t] + cumulative[after][ticks] - cumulative[after][t];
                if (loss < oracle.one_switch_loss - 1e-12) {
                    oracle.one_switch_loss = loss;
                    oracle.switch_index = t;
                    oracle.before_arm = before;
                    oracle.after_arm = after;
                }
            }
        }
    }
    return oracle;
}
