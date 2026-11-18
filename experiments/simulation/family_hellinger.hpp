#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "strategy_runs.hpp"

namespace strategy_experiments {

// Squared Hellinger distance for mixtures with non-Gaussian working components.
class FamilyHellinger {
public:
    FamilyHellinger(thesis::H2Table& h2, std::span<const FamilyInverse, 4> inverses, bool enabled)
        : h2_(h2) {
        if (!enabled) return;
        const auto& grid = h2_.g;
        const int size = grid.size();
        for (std::size_t family = 1; family < inverses.size(); ++family) {
            auto& table = tables_[family];
            table.resize(std::size_t(kPoints) * size * size);
            for (int a = 0; a < kPoints; ++a) {
                const auto& inverse = inverses[family];
                const double parameter = inverse(kMin + kStep * a);
                for (int i = 0; i < size; ++i)
                    for (int j = 0; j < size; ++j)
                        table[(std::size_t(a) * size + i) * size + j] =
                            float(density(inverse.F, parameter, grid.u[i], grid.u[j]));
            }
        }
    }

    double single_working_family(int family, double parameter, const double* rho,
                                 const double* weights, std::size_t count) {
        if (family == 0) return h2_.mix(parameter, rho, weights, count);
        return mixture(h2_.F, parameter, rho, weights, count, family);
    }

    double mixed_families(copula::Family truth_family, double parameter, const int* families,
                          const double* rho, const double* weights, std::size_t count) {
        return mixture(truth_family, parameter, rho, weights, count, 0, families);
    }

private:
    static constexpr int kPoints = 799;
    static constexpr double kMin = -0.995, kStep = 0.0025;

    struct Component {
        const float* lower = nullptr;
        double fraction = 0, rho = 0, normalizer = 0, exponent = 0;
    };

    static double density(copula::Family family, double parameter, double u, double v) {
        const double log_density = copula::log_density(family, u, v, parameter);
        return std::isfinite(log_density) ? std::exp(log_density) : 0.0;
    }

    double mixture(copula::Family truth_family, double parameter, const double* rho,
                   const double* weights, std::size_t count, int family, const int* families = nullptr) {
        const auto& grid = h2_.g;
        const int size = grid.size();
        const std::size_t cells = std::size_t(size) * size;
        auto& truth = truth_family == h2_.F ? h2_.truth[parameter] : switched_truth_[{truth_family, parameter}];
        if (truth.empty()) {
            truth = dmetric::Table(grid, [&](double u, double v) {
                return density(truth_family, parameter, u, v);
            }).v;
        }

        std::vector<Component> components(count);
        for (std::size_t k = 0; k < count; ++k) {
            auto& component = components[k];
            const int working = families ? families[k] : family;
            if (working == 0) {
                const double r = copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rho[k], -0.999, 0.999));
                component.rho = r;
                component.normalizer = 1 / std::sqrt(1 - r * r);
                component.exponent = 1 / (2 * (1 - r * r));
            } else {
                const double position = (std::clamp(rho[k], kMin, kMin + kStep * (kPoints - 1)) - kMin) / kStep;
                const auto lower = std::min(std::size_t(kPoints - 2), std::size_t(position));
                component.lower = tables_[working].data() + lower * cells;
                component.fraction = position - lower;
            }
        }

        return 1.0 - dmetric::integrate(grid, [&](int i, int j) {
            const double x = grid.z[i], y = grid.z[j];
            const std::size_t cell = std::size_t(i) * size + j;
            double mixture_density = 0;
            for (std::size_t k = 0; k < count; ++k) {
                if (weights[k] <= 0) continue;
                const auto& c = components[k];
                if (c.lower) {
                    mixture_density += weights[k] * ((1 - c.fraction) * c.lower[cell] + c.fraction * c.lower[cell + cells]);
                } else {
                    const double r = c.rho;
                    mixture_density += weights[k] * c.normalizer *
                        std::exp(-(r * r * (x * x + y * y) - 2 * r * x * y) * c.exponent);
                }
            }
            return std::sqrt(std::max(truth[cell], 0.0) * mixture_density);
        });
    }

    thesis::H2Table& h2_;
    std::array<std::vector<float>, 4> tables_;
    std::map<std::pair<copula::Family, double>, std::vector<double>> switched_truth_;
};

}  // namespace strategy_experiments
