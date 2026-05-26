#include <App.h>

#include <iostream>
#include <thread>
#include <chrono>

#include "../streamgenerator/datasource/synthetic_source.hpp"
#include "../streamgenerator/protocol/serializer.hpp"

struct PerSocketData {};

int main() {
    uWS::App app;

    SyntheticSource source;
    Serializer serializer;

    // ---------------------------
    // WebSocket setup
    // ---------------------------
    app.ws<PerSocketData>("/*", {
        .open = [](auto *ws) {
            std::cout << "client connected\n";
            ws->subscribe("ticks");
        },
        .close = [](auto *ws, int, std::string_view) {
            std::cout << "client disconnected\n";
        }
    });

    // ---------------------------
    // Stream producer
    // ---------------------------
    std::thread producer([&]() {
        while (true) {
            auto event = source.next();
            auto msg = serializer.to_json(event);

            app.publish("ticks", msg, uWS::OpCode::TEXT);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // ---------------------------
    // Start server
    // ---------------------------
    app.listen(9002, [](auto *token) {
        if (token) {
            std::cout << "listening on 9002\n";
        }
    });

    app.run();

    producer.join();
}