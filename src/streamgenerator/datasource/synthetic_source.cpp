#include "synthetic_source.hpp"

#include <cmath>
#include <chrono>

SyntheticSource::SyntheticSource()
    : price_x_(100.0),
      price_y_(100.0),
      volume_x_(1000.0),
      volume_y_(1000.0),
      drift_x_(0.0001),
      drift_y_(0.0001),
      vol_x_(0.01),
      vol_y_(0.01),
      rng_(std::random_device{}()),
      normal_(0.0, 1.0)
{
    corr_ <<
        1.0, 0.6, -0.2, -0.2,
        0.6, 1.0, -0.2, -0.2,
       -0.2,-0.2,  1.0,  0.4,
       -0.2,-0.2,  0.4,  1.0;
}

Eigen::Vector4d SyntheticSource::copula_sample() {
    Eigen::Vector4d z;

    for (int i = 0; i < 4; ++i) {
        z(i) = normal_(rng_);
    }

    Eigen::Matrix4d L = corr_.llt().matrixL();

    return L * z;
}

MarketEvent SyntheticSource::next() {
    auto z = copula_sample();

    double r_x = drift_x_ + vol_x_ * z(0);
    double r_y = drift_y_ + vol_y_ * z(1);

    price_x_ *= std::exp(r_x);
    price_y_ *= std::exp(r_y);

    auto now = std::chrono::system_clock::now();

    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    return MarketEvent{
        price_x_,
        price_y_,
        volume_x_,
        volume_y_,
        r_x,
        r_y,
        ts
    };
}