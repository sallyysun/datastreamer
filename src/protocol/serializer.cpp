#include "serializer.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string Serializer::to_json(const SymbolTick& tick) {
    json j{
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

SymbolTick Serializer::symbol_from_json(const std::string& text) {
    json j = json::parse(text);
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
