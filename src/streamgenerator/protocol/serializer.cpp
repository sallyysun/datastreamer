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
        {"timestamp", event.timestamp}
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
