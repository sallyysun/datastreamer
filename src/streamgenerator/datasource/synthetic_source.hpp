#pragma once

#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "copula.hpp"
#include "datasource.hpp"
#include "dependence.hpp"

// Synthetic bivariate market: log returns drift + vol * (copula draw), so the two prices' dependence follows the
// scenario's Spearman rho and the copula family. Each tick carries the ground truth when a family name is given.
class SyntheticSource : public DataSource {
public:
    explicit SyntheticSource(std::unique_ptr<Dependence> dependence = std::make_unique<ConstantDependence>(0.6),
                             std::unique_ptr<dgp::Copula> copula = std::make_unique<dgp::GaussianCopula>(),
                             unsigned seed = std::random_device{}(), std::string family = "")
        : dependence_(std::move(dependence)), copula_(std::move(copula)), rng_(seed), family_(std::move(family)) {}

    TickPair next() override {
        if (!opened_) {   // the opening prices, before the first return
            opened_ = true;
            const long long ts0 = now_ms();
            return TickPair{SymbolTick{"", price_x_, volume_x_, ts0, 0.0}, SymbolTick{"", price_y_, volume_y_, ts0, 0.0}};
        }
        const auto sample = CopulaSource::draw(*dependence_, *copula_, rng_);
        price_x_ *= std::exp(drift_x_ + vol_x_ * sample.x);
        price_y_ *= std::exp(drift_y_ + vol_y_ * sample.y);
        const long long ts = now_ms();
        SymbolTick x{"", price_x_, volume_x_, ts, sample.true_spearman};   // the server assigns the symbols
        SymbolTick y{"", price_y_, volume_y_, ts, sample.true_spearman};
        if (!family_.empty()) {
            x.true_pit = special::normal_cdf(sample.x);
            y.true_pit = special::normal_cdf(sample.y);
            x.true_target = y.true_target = sample.target_spearman;
            x.true_family = y.true_family = family_;
        }
        return TickPair{x, y};
    }

    // Change the generating copula family from the next draw on (same dependence path and generator).
    void set_family(const std::string& family) { copula_ = dgp::make_copula(family); family_ = family; }

private:
    static long long now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    double price_x_ = 100.0, price_y_ = 100.0;
    double volume_x_ = 1000.0, volume_y_ = 1000.0;
    double drift_x_ = 0.0001, drift_y_ = 0.0001;
    double vol_x_ = 0.01, vol_y_ = 0.01;
    std::unique_ptr<Dependence> dependence_;
    std::unique_ptr<dgp::Copula> copula_;
    std::mt19937 rng_;
    bool opened_ = false;
    std::string family_;
};
