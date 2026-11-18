#pragma once

#include <string>

#include <nlohmann/json.hpp>

// One symbol's price and volume at a given time, with ground truth for synthetic streams.
// Consumers pair synchronous ticks and derive log returns from prices.
struct SymbolTick {
    std::string symbol;
    double price = 0.0;
    double volume = 0.0;
    long long timestamp = 0;
    double true_rho_s = 0.0;
    double true_pit = 0.0;
    double true_knob = 0.0;
    std::string true_family;
};

struct TickPair {
    SymbolTick x;
    SymbolTick y;
};

inline std::string equity_topic(const std::string& symbol) {
    return "trades.equity." + symbol;
}

// Match "all", an exact topic, or a dot-delimited prefix.
inline bool topic_matches(const std::string& sub, const std::string& pub) {
    if (sub == "all" || sub == pub) return true;
    return pub.size() > sub.size() && pub[sub.size()] == '.' &&
           pub.compare(0, sub.size(), sub) == 0;
}

class Serializer {
public:
    static std::string to_json(const SymbolTick& tick);
    static SymbolTick symbol_from_json(const std::string& text);
};

inline std::string Serializer::to_json(const SymbolTick& tick) {
    nlohmann::json j{
        {"symbol", tick.symbol},
        {"price", tick.price},
        {"volume", tick.volume},
        {"timestamp", tick.timestamp},
        {"true_rho_s", tick.true_rho_s}
    };
    if (!tick.true_family.empty()) {
        j["true_pit"] = tick.true_pit;
        j["true_knob"] = tick.true_knob;
        j["true_family"] = tick.true_family;
    }
    return j.dump();
}

inline SymbolTick Serializer::symbol_from_json(const std::string& text) {
    nlohmann::json j = nlohmann::json::parse(text);
    SymbolTick tick;
    tick.symbol = j.at("symbol").get<std::string>();
    tick.price = j.at("price").get<double>();
    tick.volume = j.at("volume").get<double>();
    tick.timestamp = j.at("timestamp").get<long long>();
    if (j.contains("true_rho_s")) tick.true_rho_s = j["true_rho_s"].get<double>();
    if (j.contains("true_family")) {
        tick.true_pit = j["true_pit"].get<double>();
        tick.true_knob = j["true_knob"].get<double>();
        tick.true_family = j["true_family"].get<std::string>();
    }
    return tick;
}
