#include <App.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

struct PerSocketData {};

int main() {
    uWS::App app;

    int tick = 0;

    app.ws<PerSocketData>("/*", {
        .open = [](auto *ws) {
            std::cout << "client connected\n";
            ws->subscribe("ticks");
        },

        .close = [](auto *ws, int code, std::string_view message) {
            std::cout << "client disconnected\n";
        }
    });

    // Run a background thread to generate data
    std::thread producer([&]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            std::string msg = "{\"type\":\"tick\",\"value\":" + std::to_string(tick++) + "}";

            app.publish("ticks", msg, uWS::OpCode::TEXT);

            std::cout << "sent: " << msg << std::endl;
        }
    });

    app.listen(9002, [](auto *listen_socket) {
        if (listen_socket) {
            std::cout << "listening on port 9002\n";
        } else {
            std::cout << "failed to listen\n";
        }
    });

    app.run();

    producer.join();
}