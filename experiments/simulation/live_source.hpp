#pragma once

// LIVE=1 starts a seeded server per stream and reconstructs returns from TCP ticks.

#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../../src/consumer/broker_client.hpp"

struct LiveStream { std::vector<double> x, y, rho_s, pu, pv, d; };

inline bool live_mode() { return std::getenv("LIVE") != nullptr; }

namespace live_detail {
inline int free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    socklen_t len = sizeof(a);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0 || ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) < 0) {
        ::close(fd); return -1;
    }
    const int p = ntohs(a.sin_port);
    ::close(fd);
    return p;
}
}  // namespace live_detail

// Starts a server with these extra environment variables, reads n observations over TCP and rebuilds them as the
// consumer does. family = the expected true_family on the ticks ("" for replayed real data).
inline LiveStream live_run(const std::vector<std::pair<std::string, std::string>>& env, const std::string& family,
                           std::size_t n) {
    const char* srv = std::getenv("LIVE_SERVER");
    const std::string server = srv ? srv : "../../server";
    for (int attempt = 0; attempt < 10; ++attempt) {
        const int bport = live_detail::free_port(), wport = live_detail::free_port();
        if (bport < 0 || wport < 0 || bport == wport) continue;
        const pid_t pid = ::fork();
        if (pid == 0) {
            for (const auto& [k, v] : env) ::setenv(k.c_str(), v.c_str(), 1);
            ::setenv("N", std::to_string(n).c_str(), 1);
            ::setenv("RATE", "0", 1); ::setenv("WAIT_CONSUMERS", "1", 1);
            ::setenv("BROKER_PORT", std::to_string(bport).c_str(), 1); ::setenv("PORT", std::to_string(wport).c_str(), 1);
            if (std::FILE* f = std::freopen("/dev/null", "w", stdout)) (void)f;
            if (std::FILE* f = std::freopen("/dev/null", "w", stderr)) (void)f;
            ::execl(server.c_str(), server.c_str(), static_cast<char*>(nullptr));
            std::_Exit(127);
        }
        BrokerClient c;
        bool up = false;
        for (int i = 0; i < 500 && !up; ++i) {   // up to 5 s for the server to listen
            up = c.connect("127.0.0.1", bport, "AAPL", "MSFT");
            if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        LiveStream s;
        TickPair e;
        double px = 0, py = 0;
        bool opened = false;
        while (up && s.x.size() < n && c.next(e)) {
            if (opened && e.x.true_family != family) break;   // not our server: retry on fresh ports
            if (opened) {
                s.x.push_back(std::log(e.x.price / px)); s.y.push_back(std::log(e.y.price / py));
                s.rho_s.push_back(e.x.true_rho_s); s.pu.push_back(e.x.true_pit); s.pv.push_back(e.y.true_pit);
                s.d.push_back(e.x.true_knob);
            }
            px = e.x.price; py = e.y.price; opened = true;
        }
        c.close();
        if (s.x.size() < n) ::kill(pid, SIGTERM);
        int st = 0;
        ::waitpid(pid, &st, 0);
        if (s.x.size() == n) return s;
    }
    throw std::runtime_error("live stream: no complete stream from " + server);
}

// One synthetic stream (family, scenario, seed) from a seeded server.
inline LiveStream live_stream(const std::string& cop, const std::string& sc, unsigned seed, std::size_t n) {
    return live_run({{"SEED", std::to_string(seed)}, {"COPULA", cop}, {"DEPENDENCE", sc}}, cop, n);
}

// Recorded returns (CSV date,x,y) replayed by the server; returns arrive as 0.01 x (ReplaySource's price scale).
inline LiveStream live_replay(const std::string& csv, std::size_t n) {
    return live_run({{"REPLAY", csv}}, "", n);
}
