#pragma once

#include "copula.hpp"
#include "datasource.hpp"
#include "dependence.hpp"

#include <memory>
#include <random>

// Synthetic bivariate market source. The dependence between the two return
// series is driven by a Dependence (constant or time-varying: drift / jump /
// oscillating / stochastic), and its SHAPE by a Copula family (Gaussian /
// Clayton / Gumbel / Student-t). Each event carries the true Spearman's rho for
// that step, giving online estimators a ground truth to track live.
class SyntheticSource : public DataSource {
public:
    // Defaults to a constant correlation of 0.6 with the Gaussian copula (the
    // previous fixed behaviour).
    explicit SyntheticSource(std::unique_ptr<Dependence> dependence =
                                 std::make_unique<ConstantDependence>(0.6),
                             std::unique_ptr<dgp::Copula> copula =
                                 std::make_unique<dgp::GaussianCopula>());

    TickPair next() override;

private:
    double price_x_;
    double price_y_;

    double volume_x_;
    double volume_y_;

    double drift_x_;
    double drift_y_;

    double vol_x_;
    double vol_y_;

    std::unique_ptr<Dependence> dependence_;
    std::unique_ptr<dgp::Copula> copula_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};
