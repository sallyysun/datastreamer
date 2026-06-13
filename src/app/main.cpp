#include <App.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include "../streamgenerator/datasource/synthetic_source.hpp"
#include "../streamgenerator/runtime/producer.hpp"
#include "../streamgenerator/runtime/event_bus.hpp"
#include "../streamgenerator/protocol/serializer.hpp"
#include "../streamgenerator/timing/delay_strategies.hpp"

#include "../device/consumer.hpp"

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

// -------------------------
// MIME TYPE HELPER
// -------------------------
static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

int main() {
    uWS::App app;

    auto bus = std::make_shared<EventBus>();
    auto source = std::make_shared<SyntheticSource>();
    Serializer serializer;

    Producer producer(source, fixed_delay(0.5)); // adjust speed here

    int connection_count = 0;

    // =========================================================
    // SERVE REACT (VITE BUILD) - IMPORTANT FIX
    // =========================================================
    app.get("/*", [](auto *res, auto *req) {
        std::string path = std::string(req->getUrl());

        // SPA fallback
        if (path == "/") {
            path = "/index.html";
        }

        std::string filePath = "client/dist" + path;

        std::ifstream file(filePath, std::ios::binary);

        // fallback for React Router (important)
        if (!file.is_open()) {
            std::ifstream fallback("client/dist/index.html");
            if (!fallback.is_open()) {
                res->writeStatus("500 Internal Server Error");
                res->end("frontend not built (client/dist missing)");
                return;
            }

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

            const ConsumerPolicy policy = next_policy(connection_count++);
            auto consumer = std::make_shared<Consumer>(1024, policy);

            EventBus::SubId id = bus->subscribe(
                EventType::All,
                [consumer](const MarketEvent& event) {
                    try {
                        consumer->on_event(event);
                    } catch (const ConsumerLagException& e) {
                        std::cerr << "consumer lag: " << e.what() << "\n";
                    }
                }
            );

            consumer->set_subscription(id);
            ws->getUserData()->consumer = consumer;

            std::cout << "client connected\n";
        },

        .close = [&](auto *ws, int, std::string_view) {
            auto &consumer = ws->getUserData()->consumer;
            if (consumer) {
                bus->unsubscribe(consumer->subscription());
            }

            std::cout << "client disconnected\n";
        }
    });

    // =========================================================
    // PRODUCER THREAD
    // =========================================================
    std::thread producer_thread([&]() {
        std::size_t n = 0;

        producer.run([&](const MarketEvent& event) {
            bus->publish(event);

            std::string msg = serializer.to_json(event);
            app.publish("ticks", msg, uWS::OpCode::TEXT);

            if (++n % 20 == 0) {
                std::cout << "[producer] events=" << n
                          << " subscribers=" << bus->subscriber_count()
                          << "\n";
            }
        });
    });

    // =========================================================
    // START SERVER
    // =========================================================
    app.listen(9002, [](auto *token) {
        if (token) {
            std::cout << "server running on http://localhost:9002\n";
        } else {
            std::cout << "failed to bind port 9002\n";
        }
    });

    app.run();
    producer_thread.join();
}