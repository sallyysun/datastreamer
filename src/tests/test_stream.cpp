#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include "../streamgenerator/buffer/ring_buffer.hpp"
#include "../streamgenerator/runtime/consumer.hpp"
#include "../streamgenerator/event/market_event.hpp"
    
MarketEvent make_dummy(int i) {
    return MarketEvent{
        100.0 + i,
        200.0 + i,
        1000,
        2000,
        0.1,
        -0.1,
        i
    };
}

int main() {
    auto buffer = std::make_shared<RingBuffer>(100);

    Consumer fast(buffer);
    Consumer slow(buffer);

    // -------------------------
    // PRODUCER THREAD
    // -------------------------
    std::thread producer([&]() {
        for (int i = 0; i < 20; i++) {
            auto event = make_dummy(i);

            buffer->push(event);
            std::cout << "[producer] pushed " << i << "\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // -------------------------
    // FAST CONSUMER
    // -------------------------
    std::thread c1([&]() {
        while (true) {
            while (fast.has_next()) {
                auto e = fast.next();
                std::cout << "FAST got price_x=" << e.price_x << "\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // -------------------------
    // SLOW CONSUMER
    // -------------------------
    std::thread c2([&]() {
        while (true) {
            while (slow.has_next()) {
                auto e = slow.next();
                std::cout << "    SLOW got price_x=" << e.price_x << "\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    });

    producer.join();
    c1.join();
    c2.join();
}