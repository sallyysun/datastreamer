#include <App.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "../streamgenerator/datasource/synthetic_source.hpp"
#include "../streamgenerator/runtime/producer.hpp"
#include "../streamgenerator/runtime/event_bus.hpp"
#include "../streamgenerator/protocol/serializer.hpp"
#include "../streamgenerator/timing/delay_strategies.hpp"

#include "../device/consumer.hpp"
#include "../device/dependence_monitor.hpp"

using json = nlohmann::json;

// =========================================================
// CONFIG LOADER
// =========================================================
struct Config {
    int port = 9002;
};

static Config load_config() {
    std::ifstream f("config.json");
    if (!f.is_open()) return {};

    json j;
    f >> j;

    return {
        .port = j["server"]["port"]
    };
}

// =========================================================
// MIME TYPE HELPER (PUT HERE)
// =========================================================
static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

// =========================================================
// SOCKET STATE
// =========================================================
struct PerSocketData {
    std::shared_ptr<Consumer> consumer;
};

static ConsumerPolicy next_policy(int n) {
    switch (n % 3) {
        case 0: return ConsumerPolicy::ProcessAll;
        case 1: return ConsumerPolicy::Snapshot;
        default: return ConsumerPolicy::SkipToLatest;
    }
}

int main() {
    Config cfg = load_config();

    uWS::App app;

    auto bus = std::make_shared<EventBus>();
    auto source = std::make_shared<SyntheticSource>();
    Serializer serializer;

    // Server-side online dependence estimators over the shared stream. Updated
    // once per event in the producer thread, so no locking is required.
    DependenceMonitor monitor;

    Producer producer(source, fixed_delay(10.0));

    int connection_count = 0;

    // =========================================================
    // SERVE REACT BUILD
    // =========================================================
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

    // =========================================================
    // WEBSOCKET STREAM
    // =========================================================
    app.ws<PerSocketData>("/*", {

        .open = [&](auto *ws) {
            ws->subscribe("ticks");

            auto consumer = std::make_shared<Consumer>(1024, next_policy(connection_count++));

            EventBus::SubId id = bus->subscribe(
                EventType::All,
                [consumer](const MarketEvent &event) {
                    consumer->on_event(event);
                }
            );

            consumer->set_subscription(id);
            ws->getUserData()->consumer = consumer;

            std::cout << "client connected\n";
        },

        .close = [&](auto *ws, int, std::string_view) {
            auto &c = ws->getUserData()->consumer;
            if (c) bus->unsubscribe(c->subscription());
        }
    });

    // =========================================================
    // PRODUCER THREAD
    // =========================================================
    std::thread producer_thread([&]() {
        std::size_t n = 0;

        producer.run([&](const MarketEvent &event) {
            bus->publish(event);

            // Fold the event into the live estimators, then ship the tick with
            // the current dependence estimates attached.
            monitor.update(event);

            std::string msg = serializer.to_json(event, monitor.snapshot());
            app.publish("ticks", msg, uWS::OpCode::TEXT);

            if (++n % 20 == 0) {
                std::cout << "events=" << n
                          << " subscribers=" << bus->subscriber_count()
                          << "\n";
            }
        });
    });

    // =========================================================
    // START SERVER
    // =========================================================
    app.listen(cfg.port, [&](auto *token) {
        if (token)
            std::cout << "server running on port " << cfg.port << "\n";
        else
            std::cout << "failed to bind port\n";
    });

    app.run();
    producer_thread.join();
}