// Server: synthetic DGP producer publishing per-symbol ticks over the TCP broker and uWS WebSocket.
// Does not estimate; estimation lives in each consumer (src/consumer). See notes/architecture.md.
#include <App.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include <vector>

#include "datasource/synthetic_source.hpp"
#include "runtime/producer.hpp"
#include "runtime/broker.hpp"
#include "../protocol/serializer.hpp"
#include "../protocol/topic.hpp"
#include "runtime/delay_strategies.hpp"

using json = nlohmann::json;

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

static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

static std::pair<std::string, std::string> live_symbols() {
    const char *env = std::getenv("SYMBOLS");
    const std::string s = env && *env ? env : "AAPL,MSFT";
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos) return {s, s};
    return {s.substr(0, comma), s.substr(comma + 1)};
}

static std::string equity_topic(const std::string &symbol) {
    return "trades.equity." + symbol;
}

static std::unique_ptr<Dependence> make_live_dependence() {
    const char* env = std::getenv("DEPENDENCE");
    const std::string name = env ? env : "oscillate";
    if (name == "constant")   return std::make_unique<ConstantDependence>(0.6);
    if (name == "drift")      return std::make_unique<DriftDependence>(-0.5, 0.8, 1500);
    if (name == "jump")       return std::make_unique<JumpDependence>(0.2, 0.8, 1000);
    if (name == "stochastic") return std::make_unique<StochasticDependence>(0.01, 0.6, 12345);
    if (name != "oscillate") std::cout << "unknown DEPENDENCE='" << name << "', using oscillate\n";
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);  // period 400 = offline oscillate, so RMSEs compare
}

static std::unique_ptr<dgp::Copula> make_live_copula() {
    const char* env = std::getenv("COPULA");
    return dgp::make_copula(env ? env : "gaussian");
}

int main() {
    Config cfg = load_config();

    uWS::App app;
    Serializer serializer;

    auto dependence = make_live_dependence();
    auto copula = make_live_copula();
    const auto symbols = live_symbols();
    const std::string sym_x = symbols.first;
    const std::string sym_y = symbols.second;
    std::cout << "live dependence: " << dependence->name() << " | copula: " << copula->name()
              << " | symbols: " << sym_x << ", " << sym_y << " (topics " << equity_topic(sym_x)
              << " / " << equity_topic(sym_y) << ")\n";
    auto source = std::make_shared<SyntheticSource>(std::move(dependence), std::move(copula));

    Broker broker(cfg.broker_port);
    if (broker.start()) {
        std::cout << "broker listening on tcp/" << cfg.broker_port << "\n";
    } else {
        std::cout << "failed to start broker on tcp/" << cfg.broker_port << "\n";
    }

    Producer producer(source, fixed_delay(10.0));

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

    const std::vector<std::string> catalog_topics = {equity_topic(sym_x), equity_topic(sym_y)};
    const std::string catalog_msg =
        json{{"type", "catalog"}, {"symbols", json::array({sym_x, sym_y})}}.dump();

    // Browser sends {"action":"subscribe"|"unsubscribe","topics":[...]}; each (possibly a dot-prefix)
    // is expanded against the catalog so the socket joins only concrete symbol topics.
    app.ws<int>("/*", {
        .open = [&catalog_msg](auto *ws) {
            ws->send(catalog_msg, uWS::OpCode::TEXT);
        },
        .message = [&catalog_topics](auto *ws, std::string_view message, uWS::OpCode) {
            json j = json::parse(message, nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.contains("action") || !j.contains("topics")) return;
            const std::string action = j["action"].get<std::string>();
            for (const auto &t : j["topics"]) {
                const std::string req = t.get<std::string>();
                for (const auto &concrete : catalog_topics) {
                    if (!topic_matches(req, concrete)) continue;
                    if (action == "subscribe") ws->subscribe(concrete);
                    else if (action == "unsubscribe") ws->unsubscribe(concrete);
                }
            }
        }
    });

    std::thread producer_thread([&]() {
        std::size_t n = 0;
        producer.run([&](const TickPair &event) {
            SymbolTick tick_x = event.x; tick_x.symbol = sym_x;
            SymbolTick tick_y = event.y; tick_y.symbol = sym_y;
            const std::string msg_x = serializer.to_json(tick_x);
            const std::string msg_y = serializer.to_json(tick_y);
            broker.publish(equity_topic(sym_x), msg_x);
            broker.publish(equity_topic(sym_y), msg_y);
            app.publish(equity_topic(sym_x), msg_x, uWS::OpCode::TEXT);
            app.publish(equity_topic(sym_y), msg_y, uWS::OpCode::TEXT);
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
