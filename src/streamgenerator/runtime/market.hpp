#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../consumer/live_lab.hpp"
#include "../../protocol.hpp"
#include "demo_config.hpp"
#include "../datasource/dependence.hpp"
#include "../datasource/file_source.hpp"

namespace market {

using nlohmann::json;
namespace fs = std::filesystem;

struct ScheduledEvent {
    double target;
    std::size_t duration;
};

class EventSchedule {
public:
    explicit EventSchedule(std::string path) : path_(std::move(path)) {}

    void configure(const DemoConfig& demo) {
        first_ = demo.market_first_event;
        interval_ = demo.market_event_interval;
        loop_ = demo.market_loop;
        builtin_ = demo.market_events;
    }

    std::optional<ScheduledEvent> at(std::size_t step) {
        if (path_.empty()) {
            if (builtin_.empty() || step < first_ || (step - first_) % interval_ != 0) return std::nullopt;
            const auto index = (step - first_) / interval_;
            if (!loop_ && index >= builtin_.size()) return std::nullopt;
            const auto& event = builtin_[index % builtin_.size()];
            return ScheduledEvent{event[0], std::size_t(event[1])};
        }
        std::error_code ec;
        const auto mtime = fs::last_write_time(path_, ec);
        if (!ec && mtime_ != mtime) reload(mtime);
        const auto found = events_.find(step);
        return found == events_.end() ? std::nullopt : std::optional{found->second};
    }

private:
    void reload(fs::file_time_type mtime) {
        std::ifstream in(path_);
        if (!in) return;
        mtime_ = mtime;
        events_.clear();
        for (std::string line; std::getline(in, line);) {
            if (line.empty() || line.front() == '#' || line.starts_with("observation,")) continue;
            std::istringstream fields(line);
            long long step, duration;
            double target;
            char a, b;
            if (fields >> step >> a >> target >> b >> duration && a == ',' && b == ',' &&
                step > 0 && target >= 0 && target <= 0.95 && duration >= 0) {
                events_[step] = {target, std::size_t(duration)};
            } else {
                std::cerr << "Market skipped invalid event row: " << line << '\n';
            }
        }
    }

    std::string path_;
    std::size_t first_ = 0, interval_ = 1;
    bool loop_ = true;
    std::vector<std::array<double, 2>> builtin_;
    std::optional<fs::file_time_type> mtime_;
    std::map<std::size_t, ScheduledEvent> events_;
};

struct MarketFrame {
    std::vector<double> values;
    std::vector<double> loadings;
    std::string date;
    std::string event;
    std::vector<double> returns;
    std::size_t step = 0;
    long long timestamp = 0;
};

class Simulation {
public:
    explicit Simulation(unsigned seed, std::string events) : events_(std::move(events)), rng_(seed) {}

    void reseed(unsigned seed) { rng_.seed(seed); }
    void configure(const DemoConfig& demo) {
        events_.configure(demo);
        dependence_ = ControlledDependence(demo.market_start_rho);
    }

    MarketFrame next(std::size_t step) {
        MarketFrame frame;
        if (auto event = events_.at(step)) {
            if (event->duration) dependence_.drift(event->target, event->duration);
            else dependence_.jump(event->target);
            char label[96];
            if (event->duration)
                std::snprintf(label, sizeof label, "Market drift to %.2f over %zu observations", event->target, event->duration);
            else
                std::snprintf(label, sizeof label, "Market jump to %.2f", event->target);
            frame.event = label;
        }
        const double factor = normal_(rng_);
        const double rho = copula::param_from_spearman(copula::Family::Gaussian, dependence_.next());
        const double scale = std::sqrt(std::abs(rho));
        // A joint Gaussian return model with a shared shock and an independent shock per asset.
        for (std::size_t i = 0; i < prices_.size(); ++i) {
            const double sign = rho < 0 && i % 2 == 1 ? -1.0 : 1.0;
            const double loading = sign * scale * exposure_[i];
            const double shock = loading * factor + std::sqrt(1 - loading * loading) * normal_(rng_);
            prices_[i] *= std::exp(0.0001 + 0.01 * shock);
            frame.loadings.push_back(loading);
            frame.values.push_back(prices_[i]);
        }
        return frame;
    }

    static constexpr std::array<double, 5> opening_prices{100, 105, 95, 110, 90};

private:
    static constexpr std::array<double, 5> exposure_{1, .95, .80, .67, .52};
    std::array<double, 5> prices_ = opening_prices;
    ControlledDependence dependence_{0.0};
    EventSchedule events_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};

using Prices = std::map<std::chrono::sys_days, double>;

inline Prices load_prices(const fs::path& path) {
    using namespace std::chrono;
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line) || (line != "unix,adj" && line != "unix,adj\r"))
        throw std::runtime_error("expected unix,adj in " + path.string());
    Prices prices;
    while (std::getline(file, line)) {
        std::size_t end = 0;
        const auto timestamp = std::stoll(line, &end);
        if (end == line.size() || line[end] != ',') throw std::runtime_error("invalid timestamp in " + path.string());
        const auto price = std::stod(line.substr(end + 1));
        if (!std::isfinite(price) || price <= 0) throw std::runtime_error("invalid adjusted price in " + path.string());
        prices[floor<days>(sys_seconds{seconds{timestamp}})] = price;
    }
    return prices;
}

class Replay {
public:
    Replay(const fs::path& directory, const std::string& tickers) {
        std::vector<Prices> histories;
        std::istringstream names(tickers);
        for (std::string name; std::getline(names, name, ',');) {
            if (name.empty() || name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.^_-") != std::string::npos ||
                std::find(symbols.begin(), symbols.end(), name) != symbols.end())
                throw std::runtime_error("invalid or duplicate replay ticker");
            histories.push_back(load_prices(directory / (name + ".csv")));
            symbols.push_back(std::move(name));
        }
        if (histories.size() < 2 || histories.size() > 5)
            throw std::runtime_error("MARKET_TICKERS must specify two to five tickers");
        for (const auto& [date, price] : histories.front()) {
            MarketFrame frame;
            frame.values.push_back(price);
            for (std::size_t i = 1; i < histories.size(); ++i) {
                const auto found = histories[i].find(date);
                if (found == histories[i].end()) break;
                frame.values.push_back(found->second);
            }
            if (frame.values.size() != histories.size()) continue;
            const std::chrono::year_month_day day(date);
            char iso[16];
            std::snprintf(iso, sizeof iso, "%04d-%02u-%02u", int(day.year()), unsigned(day.month()), unsigned(day.day()));
            frame.date = iso;
            frames_.push_back(std::move(frame));
        }
        if (frames_.size() < 2) throw std::runtime_error("replay needs at least two dates shared by all tickers");
    }

    std::size_t size() const { return frames_.size(); }
    bool at_end() const { return cursor_ == frames_.size(); }
    void rewind() { cursor_ = 0; }
    MarketFrame next() { return frames_[cursor_++]; }

    std::vector<std::string> symbols;

private:
    std::vector<MarketFrame> frames_;
    std::size_t cursor_ = 0;
};

class Broadcast {
public:
    Broadcast(unsigned seed, const char* prices_dir, const char* tickers, const char* events) {
        if (prices_dir && *prices_dir) {
            replay_.emplace(prices_dir, tickers ? tickers : "");
            symbols_ = replay_->symbols;
        } else {
            simulation_.emplace(seed, events ? events : "");
            symbols_ = {"A", "B", "C", "D", "E"};
            previous_.assign(Simulation::opening_prices.begin(), Simulation::opening_prices.end());
        }
        datasketches_kll::seed(seed);
    }

    Broadcast(const fs::path& file, std::vector<std::string> symbols) : symbols_(std::move(symbols)) {
        std::ifstream input(file);
        std::string header;
        if (!std::getline(input, header) || (header != "date,x,y" && header != "date,x,y\r"))
            throw std::runtime_error("expected date,x,y in " + file.string());
        returns_.emplace(file.string());
        if (returns_->size() < 2) throw std::runtime_error("expected at least two returns in " + file.string());
        for (std::size_t i = 0; i < returns_->size(); ++i) {
            const auto [x, y] = returns_->returns(i);
            const auto& date = returns_->date(i);
            if (!std::isfinite(x) || !std::isfinite(y) || date.size() != 10 ||
                (i > 0 && returns_->date(i - 1) >= date))
                throw std::runtime_error("invalid return or date in " + file.string());
        }
    }

    json catalog(const DemoConfig& demo) const {
        return {{"symbols", symbols_},
                {"memories", memories(demo)},
                {"readyAt", warmup() + 1}};
    }

    MarketFrame next() {
        if (returns_) {
            if (step_ == returns_->size()) step_ = 0;
            const auto [x, y] = returns_->returns(step_);
            auto date = returns_->date(step_);
            return {.values = {x, y}, .date = std::move(date), .returns = {x, y}, .step = ++step_};
        }
        if (replay_ && replay_->at_end()) {
            replay_->rewind();
            previous_.clear();
            step_ = 0;
        }
        auto frame = replay_ ? replay_->next() : simulation_->next(step_ + 1);
        frame.step = ++step_;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        for (std::size_t i = 0; i < previous_.size(); ++i)
            frame.returns.push_back(std::log(frame.values[i]) - std::log(previous_[i]));
        previous_ = frame.values;
        return frame;
    }

    void reseed(unsigned seed, const DemoConfig& demo) {
        if (!simulation_) return;
        simulation_->reseed(seed);
        simulation_->configure(demo);
    }
    std::size_t length() const { return returns_ ? returns_->size() : replay_ ? replay_->size() : 0; }
    std::size_t preroll(std::size_t wanted) const { const auto n = length(); return n ? std::min(wanted, n / 2) : wanted; }
    std::size_t symbol_count() const { return symbols_.size(); }
    std::size_t warmup() const { return simulation_ ? LiveLab::kDemoWarmup : thesis::kMffWarmup; }
    std::vector<double> memories(const DemoConfig& demo) const {
        return simulation_ ? demo.market_memories : std::vector<double>(std::begin(thesis::kMem), std::end(thesis::kMem));
    }

private:
    std::optional<Simulation> simulation_;
    std::optional<Replay> replay_;
    std::optional<FileSource> returns_;
    std::vector<std::string> symbols_;
    std::vector<double> previous_;
    std::size_t step_ = 0;
};

class Subscription {
public:
    Subscription(Broadcast stream, std::size_t x, std::size_t y, unsigned seed, double rate, DemoConfig demo)
        : stream_(std::move(stream)), demo_(std::move(demo)), x_(x), y_(y), memories_(stream_.memories(demo_)),
          warmup_(stream_.warmup()), rate_(demo_.clamp_rate(rate)), model_(make_model()) {
        stream_.reseed(seed, demo_);
    }

    void set_rate(double rate) { rate_ = demo_.clamp_rate(rate); }

    std::vector<json> advance(double seconds) {
        std::vector<json> points;
        const double rate = emitted_ < stream_.preroll(demo_.warmup_observations) ? demo_.warmup_rate() : rate_;
        credit_ = std::min(credit_ + rate * seconds, 5.0);
        for (; credit_ >= 1.0; credit_ -= 1.0, ++emitted_) points.push_back(observe(stream_.next()));
        return points;
    }

private:
    json observe(const MarketFrame& frame) {
        if (frame.step == 1 && observations_ > 0) {
            model_ = make_model();
            observations_ = 0;
        }
        if (!frame.returns.empty()) {
            model_.update(frame.returns[x_], frame.returns[y_]);
            ++observations_;
        }
        json point = {{"type", "market"}, {"step", frame.step}, {"observations", observations_},
                      {"x", frame.values[x_]}, {"y", frame.values[y_]},
                      {"mff", nullptr}, {"weights", model_.memory_weights()}};
        if (model_.mff_ready()) point["mff"] = model_.mff();
        if (!frame.loadings.empty()) point["truth"] = copula::spearman_from_param(
            copula::Family::Gaussian, frame.loadings[x_] * frame.loadings[y_]);
        if (!frame.date.empty()) point["date"] = frame.date;
        else point["timestamp"] = frame.timestamp;
        if (!frame.event.empty()) point["event"] = frame.event;
        return point;
    }

    LiveLab make_model() const {
        return LiveLab("rmpl", {memories_.begin(), memories_.end()}, {"gaussian"}, demo_.eta, demo_.delta, warmup_);
    }

    Broadcast stream_;
    DemoConfig demo_;
    std::size_t x_, y_;
    std::vector<double> memories_;
    std::size_t warmup_;
    double rate_;
    double credit_ = 0;
    std::size_t emitted_ = 0;
    std::size_t observations_ = 0;
    LiveLab model_;
};

} // namespace market
