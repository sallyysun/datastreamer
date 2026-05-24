#pragma once

#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <memory>
#include <string>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace asio = boost::asio;
using tcp = asio::ip::tcp;

class WebSocketServer;

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(tcp::socket socket, WebSocketServer& server);

    void start();

    void send(const std::string& message);

private:
    websocket::stream<tcp::socket> ws_;
    beast::flat_buffer buffer_;
    WebSocketServer& server_;

    void do_read();
    void on_write(beast::error_code ec, std::size_t bytes_transferred);
};