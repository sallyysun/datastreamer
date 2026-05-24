#include "serializer.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string Serializer::to_json(const MarketEvent& event) {
    json j = {
        {"price_x", event.price_x},
        {"price_y", event.price_y},
        {"volume_x", event.volume_x},
        {"volume_y", event.volume_y},
        {"return_x", event.return_x},
        {"return_y", event.return_y},
        {"timestamp", event.timestamp}
    };

    return j.dump();
}