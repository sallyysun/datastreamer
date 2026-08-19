#include <App.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "../streamgenerator/datasource/synthetic_source.hpp"
#include "../streamgenerator/runtime/producer.hpp"
#include "../streamgenerator/runtime/broker.hpp"
#include "../streamgenerator/protocol/serializer.hpp"
#include "../streamgenerator/runtime/delay_strategies.hpp"

using json = nlohmann::json;

// =========================================================
// CONFIG LOADER
// =========================================================
struct Config {
    int port = 9002;        // uWS HTTP/WebSocket (browser display)
    int broker_port = 9003; // raw-TCP broker (C++ consumers)
};

static Config load_config() {
    std::ifstream f("config.json");
    if (!f.is_open()) return {};
    json j;
    f >> j;
    Config c;
    c.port = j["server"]["port"];
    if (j["server"].contains("broker_port")) c.broker_port = j["server"]["broker_port"];
    return c;
}

// =========================================================
// MIME + topic helpers
// =========================================================
static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

// Topic a MarketEvent is published on (consumers subscribe by topic).
static const char* topic_for(const MarketEvent &event) {
    return event.type == EventType::Quote ? "quotes" : "trades";
}

// Pick the live dependence regime from the DEPENDENCE env var (see
// make_live_dependence below). Same binary demos any regime without recompiling.
static std::unique_ptr<Dependence> make_live_dependence() {
    const char* env = std::getenv("DEPENDENCE");
    const std::string name = env ? env : "oscillate";
    if (name == "constant")   return std::make_unique<ConstantDependence>(0.6);
    if (name == "drift")      return std::make_unique<DriftDependence>(-0.5, 0.8, 1500);
    if (name == "jump")       return std::make_unique<JumpDependence>(0.2, 0.8, 1000);
    if (name == "stochastic") return std::make_unique<StochasticDependence>(0.01, 0.6, 12345);
    if (name != "oscillate") std::cout << "unknown DEPENDENCE='" << name << "', using oscillate\n";
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}

// =========================================================
// SERVER = stream generator (producer) + broker.
// The server does NOT estimate: it publishes raw events. Estimation lives in
// each consumer's device (see src/consumer). The browser is a display client.
// =========================================================
int main() {
    Config cfg = load_config();

    uWS::App app;
    Serializer serializer;

    auto dependence = make_live_dependence();
    std::cout << "live dependence: " << dependence->name() << "\n";
    auto source = std::make_shared<SyntheticSource>(std::move(dependence));

    // Raw-TCP broker for C++ consumers.
    Broker broker(cfg.broker_port);
    if (broker.start()) {
        std::cout << "broker listening on tcp/" << cfg.broker_port << "\n";
    } else {
        std::cout << "failed to start broker on tcp/" << cfg.broker_port << "\n";
    }

    Producer producer(source, fixed_delay(10.0));

    // Serve the React build (display client).
    app.get("/*", [](auto *res, auto *req) {
        std::string path = std::string(req->getUrl());
        if (path == "/") path = "/index.html";
        std::string filePath = "client/dist" + path;
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::ifstream fallback("client/dist/index.html");
            std::stringstream buffer;
            buffer << fallback.rdbuf();
            res->writeHeader("Content-Type", "text/html");
            res->end(buffer.str());
            return;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        res->writeHeader("Content-Type", mime_type(filePath));
        res->end(buffer.str());
    });

    // Browser display: raw event stream over WebSocket (no server-side estimates).
    app.ws<int>("/*", {
        .open = [](auto *ws) { ws->subscribe("ticks"); }
    });

    // Producer thread: pull events, publish RAW to the broker (per topic) and to
    // the browser display topic. No estimation here.
    std::thread producer_thread([&]() {
        std::size_t n = 0;
        producer.run([&](const MarketEvent &event) {
            const std::string msg = serializer.to_json(event);
            broker.publish(topic_for(event), msg);
            app.publish("ticks", msg, uWS::OpCode::TEXT);
            if (++n % 20 == 0) {
                std::cout << "events=" << n
                          << " tcp_consumers=" << broker.subscriber_count() << "\n";
            }
        });
    });

    app.listen(cfg.port, [&](auto *token) {
        std::cout << (token ? "server running on port " : "failed to bind port ")
                  << cfg.port << "\n";
    });

    app.run();
    producer_thread.join();
    broker.stop();
}
