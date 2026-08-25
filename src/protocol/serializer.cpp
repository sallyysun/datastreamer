#include "serializer.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static json base_json(const MarketEvent& event) {
    return json{
        {"price_x", event.price_x},
        {"price_y", event.price_y},
        {"volume_x", event.volume_x},
        {"volume_y", event.volume_y},
        {"return_x", event.return_x},
        {"return_y", event.return_y},
        {"timestamp", event.timestamp},
        {"true_rho_s", event.true_rho_s}
    };
}

std::string Serializer::to_json(const MarketEvent& event) {
    return base_json(event).dump();
}

std::string Serializer::to_json(
    const MarketEvent& event,
    const std::vector<std::pair<std::string, double>>& dependence) {
    json j = base_json(event);

    json dep = json::object();
    for (const auto& [name, value] : dependence) {
        dep[name] = value;
    }
    j["dependence"] = dep;

    return j.dump();
}

MarketEvent Serializer::from_json(const std::string& text) {
    json j = json::parse(text);
    MarketEvent event{
        j.at("price_x").get<double>(),
        j.at("price_y").get<double>(),
        j.at("volume_x").get<double>(),
        j.at("volume_y").get<double>(),
        j.at("return_x").get<double>(),
        j.at("return_y").get<double>(),
        j.at("timestamp").get<long long>()
    };
    if (j.contains("true_rho_s")) event.true_rho_s = j["true_rho_s"].get<double>();
    return event;
}

std::string Serializer::to_json(const SymbolTick& tick) {
    return json{
        {"symbol", tick.symbol},
        {"price", tick.price},
        {"volume", tick.volume},
        {"return", tick.ret},
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
    tick.ret = j.at("return").get<double>();
    tick.timestamp = j.at("timestamp").get<long long>();
    if (j.contains("true_rho_s")) tick.true_rho_s = j["true_rho_s"].get<double>();
    return tick;
}
