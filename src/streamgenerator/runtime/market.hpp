#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "../../consumer/live_lab.hpp"
#include "../datasource/file_source.hpp"

namespace market {

inline constexpr char kTopic[] = "market.pair";

inline LiveLab estimator() {
    return LiveLab("rmpl", {thesis::kMem, thesis::kMem + thesis::kK}, {"gaussian"}, 1.0, 1e-4);
}

inline nlohmann::json reading(const LiveLab& model, double x, double y) {
    nlohmann::json point = {{"x", x}, {"y", y}, {"mff", nullptr}};
    if (model.mff_ready()) {
        point["mff"] = model.mff();
        point["weights"] = model.memory_weights();
    }
    return point;
}

struct Replays {
    nlohmann::json names = nlohmann::json::array();
    std::map<std::string, std::string> bodies;
};

inline std::map<std::string, std::filesystem::path> csv_files(const char* directory) {
    if (!directory || !*directory) return {};
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::directory_iterator entry(directory, ec);
    std::map<std::string, fs::path> files;
    for (; !ec && entry != fs::directory_iterator(); entry.increment(ec)) {
        std::error_code status_error;
        if (fs::is_regular_file(entry->symlink_status(status_error)) && entry->path().extension() == ".csv")
            files.emplace(entry->path().stem().string(), entry->path());
    }
    if (ec) std::cerr << "could not list " << directory << ": " << ec.message() << '\n';
    return files;
}

struct ReplayRun {
    explicit ReplayRun(std::uint64_t seed) { datasketches_kll::seed(seed); }

    void add(const std::string& date, double x, double y) {
        if (date.empty() || !std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("missing date or invalid return");
        model.update(x, y);
        auto point = reading(model, x, y);
        point["date"] = date;
        point["step"] = points.size() + 1;
        points.push_back(std::move(point));
    }

    LiveLab model = estimator();
    nlohmann::json points = nlohmann::json::array();
};

inline Replays load_replays(const char* directory, std::uint64_t seed) {
    Replays replays;
    for (const auto& [name, path] : csv_files(directory)) {
        try {
            const FileSource source(path.string());
            if (!source.size()) continue;
            ReplayRun run(seed);
            for (std::size_t i = 0; i < source.size(); ++i) {
                const auto [x, y] = source.returns(i);
                run.add(source.date(i), x, y);
            }
            const std::string id = std::to_string(replays.bodies.size());
            replays.bodies.emplace(id, run.points.dump());
            const auto separator = name.find('_');
            const bool named_pair = separator != std::string::npos && separator > 0 && separator + 1 < name.size();
            const auto labels = named_pair
                ? nlohmann::json::array({name.substr(0, separator), name.substr(separator + 1)})
                : nlohmann::json::array({"X", "Y"});
            replays.names.push_back({{"id", id}, {"name", name}, {"labels", labels}});
        } catch (const std::exception& error) {
            std::cerr << "skipping market replay " << path << ": " << error.what() << '\n';
        }
    }
    return replays;
}

using Prices = std::map<std::chrono::sys_days, double>;

struct Stocks {
    nlohmann::json names = nlohmann::json::array();
    std::map<std::string, Prices, std::less<>> histories;
};

inline Stocks load_stocks(const char* directory) {
    using namespace std::chrono;
    Stocks stocks;
    for (const auto& [symbol, path] : csv_files(directory)) {
        try {
            std::ifstream file(path);
            std::string line;
            if (!std::getline(file, line) || (line != "unix,adj" && line != "unix,adj\r")) continue;
            Prices history;
            while (std::getline(file, line)) {
                std::size_t end = 0;
                const auto timestamp = std::stoll(line, &end);
                if (end == line.size() || line[end] != ',') throw std::runtime_error("invalid price row");
                const auto price = std::stod(line.substr(end + 1));
                if (!std::isfinite(price) || price <= 0) throw std::runtime_error("invalid adjusted price");
                const auto date = floor<days>(sys_seconds{seconds{timestamp}});
                history.emplace(date, std::log(price));
            }
            if (history.size() < 2) continue;
            const std::string id = std::to_string(stocks.histories.size());
            stocks.histories.emplace(id, std::move(history));
            stocks.names.push_back({{"id", id}, {"symbol", symbol}});
        } catch (const std::exception& error) {
            std::cerr << "skipping market stock " << path << ": " << error.what() << '\n';
        }
    }
    return stocks;
}

inline nlohmann::json replay_pair(const Prices& x, const Prices& y, std::uint64_t seed) {
    ReplayRun run(seed);
    std::optional<std::pair<double, double>> previous;
    for (const auto& [date, x_log] : x) {
        const auto match = y.find(date);
        if (match == y.end()) continue;
        const double y_log = match->second;
        if (previous) {
            const std::chrono::year_month_day day(date);
            char iso[16];
            std::snprintf(iso, sizeof iso, "%04d-%02u-%02u", int(day.year()), unsigned(day.month()), unsigned(day.day()));
            run.add(iso, 100 * (x_log - previous->first), 100 * (y_log - previous->second));
        }
        previous = {x_log, y_log};
    }
    return std::move(run.points);
}

} // namespace market
