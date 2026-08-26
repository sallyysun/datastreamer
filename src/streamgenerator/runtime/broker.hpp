#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../protocol/topic.hpp"

// Minimal TCP publish/subscribe broker -- a "mini Kafka" broker. Consumers
// connect over TCP and send a line "SUBSCRIBE <topic> [max_hz]\n"; thereafter
// they receive newline-delimited JSON events published to a matching topic.
//
// Topics are hierarchical, dot-separated (e.g. "trades.SYNX_SYNY"). A
// subscription matches by prefix on dot boundaries: subscribing to "trades"
// receives every "trades.<pair>", while "trades.SYNX_SYNY" selects just that
// pair. The literal "all" matches everything.
//
// The optional [max_hz] throttles a connection to at most that many messages
// per second (0 / omitted = unthrottled); excess messages are dropped, which is
// consistent with the best-effort model below.
//
// Best-effort, NO retention or offsets by design: client sockets are
// non-blocking, so a consumer that cannot keep up simply drops events. Each
// consumer's own device buffer is its only memory -- matching the thesis's
// "buffer-type device". publish() is called from the producer thread; the
// accept/read loop runs on its own thread; a single mutex guards the
// subscription table.
class Broker {
public:
    explicit Broker(int port) : port_(port) {}
    ~Broker() { stop(); }

    bool start() {
        std::signal(SIGPIPE, SIG_IGN);  // writing to a dropped consumer must not kill us
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return false;
        int opt = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<std::uint16_t>(port_));
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) return false;
        if (::listen(listen_fd_, 16) < 0) return false;
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }

    // Deliver msg to every consumer whose subscription matches `topic`, honouring
    // each connection's optional rate limit.
    void publish(const std::string& topic, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        deliver_locked(topic, msg);
    }

    // Hook invoked for every message a CONSUMER publishes back (via "PUBLISH"),
    // e.g. an estimates.* stream. The server sets this to bridge those to the
    // browser WebSocket. Not called for producer-side publish() above.
    std::function<void(const std::string& topic, const std::string& msg)> on_message;

    std::size_t subscriber_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return subs_.size();
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (listen_fd_ >= 0) ::close(listen_fd_);
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [fd, _] : subs_) ::close(fd);
        subs_.clear();
        inbuf_.clear();
    }

private:
    void run() {
        while (running_) {
            std::vector<pollfd> pfds;
            pfds.push_back({listen_fd_, POLLIN, 0});
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const auto& [fd, _] : subs_) pfds.push_back({fd, POLLIN, 0});
            }
            if (::poll(pfds.data(), static_cast<nfds_t>(pfds.size()), 500) <= 0) continue;

            if (pfds[0].revents & POLLIN) {
                int cfd = ::accept(listen_fd_, nullptr, nullptr);
                if (cfd >= 0) {
                    ::fcntl(cfd, F_SETFL, O_NONBLOCK);
                    std::lock_guard<std::mutex> lock(mutex_);
                    subs_[cfd];   // register with an empty topic set
                    inbuf_[cfd];
                }
            }
            for (std::size_t i = 1; i < pfds.size(); ++i) {
                if (!(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
                const int fd = pfds[i].fd;
                char buf[2048];
                const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
                if (n <= 0) { drop(fd); continue; }
                handle_input(fd, std::string(buf, static_cast<std::size_t>(n)));
            }
        }
    }

    // Fan out `msg` to matching subscribers. Caller must hold mutex_.
    void deliver_locked(const std::string& topic, const std::string& msg) {
        std::string framed = msg;
        framed.push_back('\n');
        const auto now = std::chrono::steady_clock::now();
        for (auto& [fd, sub] : subs_) {
            if (!sub.matches(topic)) continue;
            if (sub.throttled(now)) continue;   // dropped: too soon since last delivery
            ::send(fd, framed.data(), framed.size(), 0);  // non-blocking: drops if full
            sub.last_sent = now;
        }
    }

    void handle_input(int fd, const std::string& chunk) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string& buf = inbuf_[fd];
        buf += chunk;
        std::size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();

            // "SUBSCRIBE <topic> [max_hz]" -- optional trailing rate limit.
            const std::string sub_kw = "SUBSCRIBE ";
            if (line.rfind(sub_kw, 0) == 0) {
                std::string rest = line.substr(sub_kw.size());
                std::string topic = rest;
                double max_hz = 0.0;
                if (const std::size_t sp = rest.find(' '); sp != std::string::npos) {
                    topic = rest.substr(0, sp);
                    max_hz = std::atof(rest.c_str() + sp + 1);
                }
                if (topic.empty()) continue;
                Sub& sub = subs_[fd];
                sub.topics.insert(topic);
                if (max_hz > 0.0) sub.min_interval = std::chrono::duration<double, std::milli>(1000.0 / max_hz);
                std::cout << "broker: fd " << fd << " subscribed to '" << topic << "'";
                if (max_hz > 0.0) std::cout << " @ " << max_hz << " msg/s";
                std::cout << "\n";
                continue;
            }

            // "PUBLISH <topic> <json>" -- a consumer publishing back (e.g. estimates).
            const std::string pub_kw = "PUBLISH ";
            if (line.rfind(pub_kw, 0) == 0) {
                const std::string rest = line.substr(pub_kw.size());
                const std::size_t sp = rest.find(' ');
                if (sp == std::string::npos) continue;
                const std::string topic = rest.substr(0, sp);
                const std::string payload = rest.substr(sp + 1);
                if (topic.empty() || payload.empty()) continue;
                deliver_locked(topic, payload);          // fan out to other consumers
                if (on_message) on_message(topic, payload);  // and bridge to the WS
                continue;
            }
        }
    }

    void drop(int fd) {
        std::lock_guard<std::mutex> lock(mutex_);
        ::close(fd);
        subs_.erase(fd);
        inbuf_.erase(fd);
    }

    // Per-connection subscription: the set of topics it wants, plus an optional
    // rate limit (min gap between deliveries; zero = unthrottled).
    struct Sub {
        using clock = std::chrono::steady_clock;
        std::unordered_set<std::string> topics;
        std::chrono::duration<double, std::milli> min_interval{0.0};
        clock::time_point last_sent{};

        // Matches if any subscribed topic matches the published one (shared
        // dot-prefix rule, see protocol/topic.hpp).
        bool matches(const std::string& pub) const {
            for (const auto& s : topics) {
                if (topic_matches(s, pub)) return true;
            }
            return false;
        }

        bool throttled(clock::time_point now) const {
            return min_interval.count() > 0.0 && (now - last_sent) < min_interval;
        }
    };

    int port_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::unordered_map<int, Sub> subs_;
    std::unordered_map<int, std::string> inbuf_;
};
