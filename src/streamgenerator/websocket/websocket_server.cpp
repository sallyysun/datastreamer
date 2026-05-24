#include "websocket_server.hpp"
#include "session.hpp"

#include <iostream>

using tcp = boost::asio::ip::tcp;

WebSocketServer::WebSocketServer(
    boost::asio::io_context& io,
    unsigned short port
) : acceptor_(io, tcp::endpoint(tcp::v4(), port)) {}

void WebSocketServer::run() {
    do_accept();
}

void WebSocketServer::do_accept() {
    acceptor_.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
            if (!ec) {
                auto session = std::make_shared<Session>(
                    std::move(socket),
                    *this
                );

                add_session(session);
                session->start();
            }

            do_accept();
        }
    );
}

void WebSocketServer::broadcast(const std::string& message) {
    for (auto& s : sessions_) {
        s->send(message);
    }
}

void WebSocketServer::add_session(std::shared_ptr<Session> session) {
    sessions_.insert(session);
}

void WebSocketServer::remove_session(std::shared_ptr<Session> session) {
    sessions_.erase(session);
}