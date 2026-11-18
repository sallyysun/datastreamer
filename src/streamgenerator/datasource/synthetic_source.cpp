#include "synthetic_source.hpp"
#include "copula_source.hpp"

#include <chrono>
#include <cmath>
#include <utility>

SyntheticSource::SyntheticSource(std::unique_ptr<Dependence> dependence,
                                 std::unique_ptr<dgp::Copula> copula,
                                 unsigned seed,
                                 std::string family)
    : price_x_(100.0),
      price_y_(100.0),
      volume_x_(1000.0),
      volume_y_(1000.0),
      drift_x_(0.0001),
      drift_y_(0.0001),
      vol_x_(0.01),
      vol_y_(0.01),
      dependence_(std::move(dependence)),
      copula_(std::move(copula)),
      rng_(seed),
      family_(std::move(family)),
      normal_(0.0, 1.0)
{
}

TickPair SyntheticSource::next() {
    // Emit the opening prices before drawing the first return.
    if (!opened_) {
        opened_ = true;
        const long long ts0 = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return TickPair{SymbolTick{"", price_x_, volume_x_, ts0, 0.0}, SymbolTick{"", price_y_, volume_y_, ts0, 0.0}};
    }
    const auto sample = CopulaSource::draw(*dependence_, *copula_, rng_);
    const double r_x = drift_x_ + vol_x_ * sample.x;
    const double r_y = drift_y_ + vol_y_ * sample.y;

    price_x_ *= std::exp(r_x);
    price_y_ *= std::exp(r_y);

    auto now = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch())
                  .count();

    // The server assigns the symbols before publishing.
    SymbolTick x{"", price_x_, volume_x_, ts, sample.rho_s};
    SymbolTick y{"", price_y_, volume_y_, ts, sample.rho_s};
    if (!family_.empty()) {
        x.true_pit = special::normal_cdf(sample.x);
        y.true_pit = special::normal_cdf(sample.y);
        x.true_target = y.true_target = sample.target;
        x.true_family = y.true_family = family_;
    }
    return TickPair{x, y};
}
