#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "../../src/copula_family.hpp"
#include "../../src/estimators/evaluation.hpp"

namespace strategy_experiments {

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

// The four working copulas the family modes compare, with their names and rho_S -> parameter tables.
struct WorkingFamilies {
    static constexpr std::size_t kCount = 4;
    static constexpr std::array<copula::Family, kCount> family{copula::Family::Gaussian, copula::Family::StudentT,
                                                               copula::Family::Clayton, copula::Family::Gumbel};
    static constexpr std::array<const char*, kCount> name{"gaussian", "t", "clayton", "gumbel"};
    const std::array<FamilyInverse, kCount> inverse{FamilyInverse(family[0]), FamilyInverse(family[1]),
                                                    FamilyInverse(family[2]), FamilyInverse(family[3])};

    // The parameter a forecast is evaluated at: the exact Gaussian conversion inside +-0.999, the table otherwise.
    double evaluation_parameter(std::size_t working, double rho) const {
        return working == 0 ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rho, -0.999, 0.999))
                            : inverse[working](rho);
    }
};

}  // namespace strategy_experiments
