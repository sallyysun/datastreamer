#include <App.h>

#include <iostream>
#include <memory>
#include <thread>

#include "../streamgenerator/datasource/synthetic_source.hpp"
#include "../streamgenerator/buffer/ring_buffer.hpp"
#include "../streamgenerator/runtime/producer.hpp"
#include "../streamgenerator/runtime/consumer_state.hpp"
#include "../streamgenerator/protocol/serializer.hpp"
#include "../streamgenerator/timing/delay_strategies.hpp"

struct PerSocketData {
    std::shared_ptr<ConsumerState> consumer;
};

int main() {
    uWS::App app;

    auto buffer = std::make_shared<RingBuffer>(1024);
    auto source = std::make_shared<SyntheticSource>();
    Serializer serializer;
    Producer producer(source, poisson_delay(10.0));

    // -------------------------
    // WebSocket
    // -------------------------
    app.ws<PerSocketData>("/*", {

        .open = [&](auto *ws) {
            ws->subscribe("ticks");

            ws->getUserData()->consumer =
                std::make_shared<ConsumerState>(buffer);

            std::cout << "client connected\n";
        },

        .close = [&](auto *, int, std::string_view) {
            std::cout << "client disconnected\n";
        }
    });

    // -------------------------
    // Producer thread
    // -------------------------
    std::thread producer_thread([&]() {
        producer.run([&](const MarketEvent& event) {
            buffer->push(event);

            std::string msg = serializer.to_json(event);

            app.publish("ticks", msg, uWS::OpCode::TEXT);

            // std::cout << msg << std::endl; // 👈 DEBUG
        });
    });

    app.listen(9002, [](auto *token) {
        if (token)
            std::cout << "server running on 9002\n";
    });

    app.run();

    producer_thread.join();
}