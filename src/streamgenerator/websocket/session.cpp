#include "session.hpp"
#include "websocket_server.hpp"

#include <iostream>

Session::Session(tcp::socket socket, WebSocketServer& server)
    : ws_(std::move(socket)),
      server_(server) {}

void Session::start() {
    ws_.accept();
    do_read();
}

void Session::do_read() {
    auto self = shared_from_this();

    ws_.async_read(buffer_,
        [self](beast::error_code ec, std::size_t) {
            if (ec) return;

            self->buffer_.consume(self->buffer_.size());
            self->do_read();
        });
}

void Session::send(const std::string& message) {
    auto self = shared_from_this();

    ws_.async_write(
        asio::buffer(message),
        [self](beast::error_code ec, std::size_t) {
            if (ec) {
                std::cerr << "Write error\n";
            }
        });
}

void Session::on_write(beast::error_code, std::size_t) {}