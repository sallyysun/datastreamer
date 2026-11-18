#pragma once

#include <cerrno>
#include <functional>

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
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../protocol.hpp"

// TCP clients send SUBSCRIBE <topic> [max_hz] and receive newline-delimited JSON.
// Delivery is best-effort by default, with no retained history.
// The broker thread accepts and reads clients. A mutex guards subscriptions during publish.
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

    void publish(const std::string& topic, const std::string& msg) {
        std::string framed = msg;
        framed.push_back('\n');
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [fd, sub] : subs_) {
            if (!sub.matches(topic)) continue;
            if (sub.throttled(now)) continue;   // dropped: too soon since last delivery
            if (lossless_) send_all(fd, framed);
            else ::send(fd, framed.data(), framed.size(), 0);  // non-blocking: drops if full
            sub.last_sent = now;
        }
    }

    // Dispatch PUBLISH <topic> <payload> on the broker thread.
    void on_publish(std::function<void(const std::string&, const std::string&)> f) { on_publish_ = std::move(f); }
    // Whether some connection subscribes to exactly this topic.
    bool subscribed(const std::string& topic) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [fd, sub] : subs_) if (sub.topics.count(topic)) return true;
        return false;
    }
    // Wait for writable sockets, giving up after a five-second stall.
    void set_lossless(bool on) { lossless_ = on; }
    // Count connections with at least min_topics subscriptions.
    std::size_t ready_count(std::size_t min_topics) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t n = 0;
        for (const auto& [fd, sub] : subs_) n += sub.topics.size() >= min_topics;
        return n;
    }
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
    static void send_all(int fd, const std::string& msg) {
        std::size_t off = 0;
        while (off < msg.size()) {
            const ssize_t k = ::send(fd, msg.data() + off, msg.size() - off, 0);
            if (k > 0) { off += static_cast<std::size_t>(k); continue; }
            if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd p{fd, POLLOUT, 0};
                if (::poll(&p, 1, 5000) <= 0) return;   // consumer stalled for 5 s: give up on this message
                continue;
            }
            return;   // connection gone
        }
    }
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

    void handle_input(int fd, const std::string& chunk) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string& buf = inbuf_[fd];
        buf += chunk;
        std::size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();

            if (line.rfind("PUBLISH ", 0) == 0) {
                const std::size_t sp = line.find(' ', 8);
                if (sp != std::string::npos && on_publish_) on_publish_(line.substr(8, sp - 8), line.substr(sp + 1));
                continue;
            }
            const std::string kw = "SUBSCRIBE ";
            if (line.rfind(kw, 0) != 0) continue;

            // "SUBSCRIBE <topic> [max_hz]" -- optional trailing rate limit.
            std::string rest = line.substr(kw.size());
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
        }
    }

    void drop(int fd) {
        std::lock_guard<std::mutex> lock(mutex_);
        ::close(fd);
        subs_.erase(fd);
        inbuf_.erase(fd);
    }

    struct Sub {
        using clock = std::chrono::steady_clock;
        std::unordered_set<std::string> topics;
        std::chrono::duration<double, std::milli> min_interval{0.0};
        clock::time_point last_sent{};

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
    bool lossless_ = false;
    std::function<void(const std::string&, const std::string&)> on_publish_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::unordered_map<int, Sub> subs_;
    std::unordered_map<int, std::string> inbuf_;
};
