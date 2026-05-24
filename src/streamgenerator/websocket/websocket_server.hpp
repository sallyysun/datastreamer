#pragma once

#include <boost/asio.hpp>
#include <memory>
#include <set>
#include <string>

class Session;

class WebSocketServer {
public:
    WebSocketServer(boost::asio::io_context& io, unsigned short port);

    void run();

    void broadcast(const std::string& message);

    void add_session(std::shared_ptr<Session> session);
    void remove_session(std::shared_ptr<Session> session);

private:
    boost::asio::ip::tcp::acceptor acceptor_;

    std::set<std::shared_ptr<Session>> sessions_;

    void do_accept();
};