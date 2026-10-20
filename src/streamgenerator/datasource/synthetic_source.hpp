#pragma once

#include "copula.hpp"
#include "datasource.hpp"
#include "dependence.hpp"

#include <memory>
#include <random>

// Synthetic bivariate GBM source: a Dependence drives the knob d, a dgp::Copula the shape. Each tick
// carries the true Spearman's rho as ground truth. Legs are unnamed; the server names them.
class SyntheticSource : public DataSource {
public:
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
