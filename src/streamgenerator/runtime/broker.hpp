#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Minimal TCP publish/subscribe broker -- a "mini Kafka" broker. Consumers
// connect over TCP and send a line "SUBSCRIBE <topic>\n"; thereafter they
// receive newline-delimited JSON events published to that topic (or the
// wildcard "all").
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

    // Deliver msg to every consumer subscribed to `topic` (or "all").
    void publish(const std::string& topic, const std::string& msg) {
        std::string framed = msg;
        framed.push_back('\n');
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [fd, topics] : subs_) {
            if (topics.count(topic) || topics.count("all")) {
                ::send(fd, framed.data(), framed.size(), 0);  // non-blocking: drops if full
            }
        }
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
            const std::string kw = "SUBSCRIBE ";
            if (line.rfind(kw, 0) == 0) {
                std::string topic = line.substr(kw.size());
                while (!topic.empty() && (topic.back() == '\r' || topic.back() == ' ')) topic.pop_back();
                subs_[fd].insert(topic);
                std::cout << "broker: fd " << fd << " subscribed to '" << topic << "'\n";
            }
        }
    }

    void drop(int fd) {
        std::lock_guard<std::mutex> lock(mutex_);
        ::close(fd);
        subs_.erase(fd);
        inbuf_.erase(fd);
    }

    int port_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::unordered_map<int, std::unordered_set<std::string>> subs_;
    std::unordered_map<int, std::string> inbuf_;
};
