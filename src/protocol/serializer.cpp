#include "serializer.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string Serializer::to_json(const SymbolTick& tick) {
    return json{
        {"symbol", tick.symbol},
        {"price", tick.price},
        {"volume", tick.volume},
        {"timestamp", tick.timestamp},
        {"true_rho_s", tick.true_rho_s}
    }.dump();
}

SymbolTick Serializer::symbol_from_json(const std::string& text) {
    json j = json::parse(text);
    SymbolTick tick;
    tick.symbol = j.at("symbol").get<std::string>();
    tick.price = j.at("price").get<double>();
    tick.volume = j.at("volume").get<double>();
    tick.timestamp = j.at("timestamp").get<long long>();
    if (j.contains("true_rho_s")) tick.true_rho_s = j["true_rho_s"].get<double>();
    return tick;
}

std::string Serializer::estimates_to_json(
    const std::string& pair, std::size_t n, double true_rho_s,
    const std::vector<std::pair<std::string, double>>& estimates) {
    json rho = json::object();
    for (const auto& [name, value] : estimates) rho[name] = value;
    return json{
        {"pair", pair},
        {"n", n},
        {"true_rho_s", true_rho_s},
        {"rho", rho}
    }.dump();
}
