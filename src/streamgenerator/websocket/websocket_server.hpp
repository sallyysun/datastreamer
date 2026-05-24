#pragma once

#include <set>
#include <string>

class WebSocketServer {
public:
    void run(unsigned short port);

    void broadcast(const std::string& message);
};