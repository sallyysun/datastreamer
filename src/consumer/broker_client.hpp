#pragma once

// TCP client of the broker: connects, subscribes to the two legs' topics ("trades.equity.<X>", "trades.equity.<Y>"),
// reads newline-delimited JSON ticks and pairs the two legs FIFO into TickPairs (producers publish both legs per step,
// in order, so front-to-front pairing is exact). Used by the consumer and by the experiments' live mode.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <deque>
#include <string>

#include "../protocol/serializer.hpp"
#include "../protocol/symbol_tick.hpp"

class BrokerClient {
public:
    // rate > 0 asks the broker to throttle this connection to that many messages per second.
    bool connect(const std::string& host, int port, const std::string& sym_x, const std::string& sym_y, double rate = 0) {
        sym_x_ = sym_x; sym_y_ = sym_y;
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0 ||
            ::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close();
            return false;
        }
        for (const std::string& s : {sym_x, sym_y}) {
            std::string sub = "SUBSCRIBE trades.equity." + s;
            if (rate > 0.0) sub += " " + std::to_string(rate);
            sub += "\n";
            ::send(fd_, sub.data(), sub.size(), 0);
        }
        return true;
    }

    // The next paired observation; false at end of stream.
    bool next(TickPair& out) {
        while (qx_.empty() || qy_.empty()) {
            const std::size_t pos = buf_.find('\n');
            if (pos == std::string::npos) {
                char chunk[65536];
                const ssize_t r = ::recv(fd_, chunk, sizeof(chunk), 0);
                if (r <= 0) return false;
                buf_.append(chunk, static_cast<std::size_t>(r));
                continue;
            }
            const std::string line = buf_.substr(0, pos);
            buf_.erase(0, pos + 1);
            if (line.empty()) continue;
            SymbolTick tick;
            try { tick = Serializer::symbol_from_json(line); } catch (...) { continue; }   // skip malformed
            if (tick.symbol == sym_x_) qx_.push_back(tick);
            else if (tick.symbol == sym_y_) qy_.push_back(tick);
        }
        out = TickPair{qx_.front(), qy_.front()};
        qx_.pop_front(); qy_.pop_front();
        return true;
    }

    // Send one line back through the broker (the lab's device state; payload must not contain a newline).
    void publish(const std::string& topic, const std::string& payload) {
        const std::string line = "PUBLISH " + topic + " " + payload + "\n";
        for (std::size_t off = 0; off < line.size();) {
            const ssize_t k = ::send(fd_, line.data() + off, line.size() - off, 0);
            if (k <= 0) return;
            off += static_cast<std::size_t>(k);
        }
    }

    void close() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }
    ~BrokerClient() { close(); }

private:
    int fd_ = -1;
    std::string sym_x_, sym_y_, buf_;
    std::deque<SymbolTick> qx_, qy_;
};
