#pragma once

// Replays recorded daily returns (CSV date,x,y; Section 6.6's EWMA-standardised returns) as two price series: the
// first tick is the opening price, then p_t = p_{t-1} exp(scale * x_t), so a device's log return ln(p_t / p_{t-1}) is
// scale * x_t. No ground truth is attached (real data).

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "datasource.hpp"

class ReplaySource : public DataSource {
public:
    explicit ReplaySource(const std::string& path, double scale = 0.01) : scale_(scale) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::string line;
        std::getline(f, line);   // header
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string date, a, b;
            std::getline(ss, date, ','); std::getline(ss, a, ','); std::getline(ss, b, ',');
            x_.push_back(std::stod(a)); y_.push_back(std::stod(b));
        }
    }
    std::size_t size() const { return x_.size(); }

    TickPair next() override {
        if (t_ > 0 && t_ <= x_.size()) {
            px_ *= std::exp(scale_ * x_[t_ - 1]);
            py_ *= std::exp(scale_ * y_[t_ - 1]);
        }
        const long long ts = static_cast<long long>(t_++);
        return TickPair{SymbolTick{"", px_, 0.0, ts, 0.0}, SymbolTick{"", py_, 0.0, ts, 0.0}};
    }

private:
    double scale_;
    std::vector<double> x_, y_;
    std::size_t t_ = 0;
    double px_ = 100.0, py_ = 100.0;
};
