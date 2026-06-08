#include <App.h>

#include <iostream>
#include <memory>
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

// Demonstrate that each consumer carries its own lag policy by rotating
// through them as clients connect.
static ConsumerPolicy next_policy(int n) {
    switch (n % 3) {
    case 0:  return ConsumerPolicy::ProcessAll;
    case 1:  return ConsumerPolicy::Snapshot;
    default: return ConsumerPolicy::SkipToLatest;
    }
}

int main() {
    uWS::App app;

    auto bus = std::make_shared<EventBus>();
    auto source = std::make_shared<SyntheticSource>();
    Serializer serializer;
    Producer producer(source, poisson_delay(10.0));

    int connection_count = 0;

    // -------------------------
    // WebSocket: one Consumer per client, each with its own ring buffer,
    // policy and StateManager, all fed through the EventBus.
    // -------------------------
    app.ws<PerSocketData>("/*", {

        .open = [&](auto *ws) {
            // raw tick stream to the client
            ws->subscribe("ticks");

            const ConsumerPolicy policy = next_policy(connection_count++);
            auto consumer = std::make_shared<Consumer>(1024, policy);

            // subscribe the consumer's buffer to the bus
            EventBus::SubId id = bus->subscribe(
                EventType::All,
                [consumer](const MarketEvent& event) {
                    try {
                        consumer->on_event(event);
                    } catch (const ConsumerLagException& e) {
                        // a ProcessAll consumer fell irrecoverably behind
                        std::cerr << "consumer lag: " << e.what() << "\n";
                    }
                });
            consumer->set_subscription(id);

            ws->getUserData()->consumer = consumer;

            std::cout << "client connected (subscribers="
                      << bus->subscriber_count() << ")\n";
        },

        .close = [&](auto *ws, int, std::string_view) {
            auto& consumer = ws->getUserData()->consumer;
            if (consumer) {
                bus->unsubscribe(consumer->subscription());
            }
            std::cout << "client disconnected (subscribers="
                      << bus->subscriber_count() << ")\n";
        }
    });

    // -------------------------
    // Producer thread: generate events, fan them out to every consumer's
    // buffer via the bus, and broadcast the raw tick to all clients.
    // -------------------------
    std::thread producer_thread([&]() {
        std::size_t n = 0;

        producer.run([&](const MarketEvent& event) {
            bus->publish(event);

            const std::string msg = serializer.to_json(event);
            app.publish("ticks", msg, uWS::OpCode::TEXT);

            // periodic peek at the stats the bus is maintaining
            if (++n % 100 == 0) {
                std::cout << "[producer] events=" << n
                          << " subscribers=" << bus->subscriber_count()
                          << "\n";
            }
        });
    });

    app.listen(9002, [](auto *token) {
        if (token)
            std::cout << "server running on 9002\n";
    });

    app.run();

    producer_thread.join();
}
