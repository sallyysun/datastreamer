#pragma once

#include "datasource.hpp"

#include <Eigen/Dense>
#include <random>

class SyntheticSource : public DataSource {
public:
    SyntheticSource();

    MarketEvent next() override;

private:
    double price_x_;
    double price_y_;

    double volume_x_;
    double volume_y_;

    double drift_x_;
    double drift_y_;

    double vol_x_;
    double vol_y_;

    Eigen::Matrix4d corr_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_;

    Eigen::Vector4d copula_sample();
};