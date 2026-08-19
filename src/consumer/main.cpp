// Standalone C++ consumer for the datastreamer mini-Kafka.
//
// Connects to the broker over TCP, subscribes to a topic, and runs its own
// device: a RingBuffer (buffer-type device), a StateManager, and an inference
// layer (the dependence estimators). Each consumer is an independent process --
// its buffer is its only memory, so a slow consumer just falls behind.
//
// Usage:
//   ./consumer [--host 127.0.0.1] [--port 9003] [--topic trades]
//              [--policy all|snapshot|latest] [--print 20]

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../device/consumer.hpp"
#include "../streamgenerator/protocol/serializer.hpp"

namespace {

ConsumerPolicy parse_policy(const std::string& s) {
    if (s == "all") return ConsumerPolicy::ProcessAll;
    if (s == "snapshot") return ConsumerPolicy::Snapshot;
    return ConsumerPolicy::SkipToLatest;  // "latest"
}

const char* arg(int argc, char** argv, const char* key, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

int connect_broker(const std::string& host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) return -1;
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) return -1;
    return fd;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host = arg(argc, argv, "--host", "127.0.0.1");
    const int port = std::atoi(arg(argc, argv, "--port", "9003"));
    const std::string topic = arg(argc, argv, "--topic", "trades");
    const ConsumerPolicy policy = parse_policy(arg(argc, argv, "--policy", "all"));
    const int print_every = std::atoi(arg(argc, argv, "--print", "20"));

    const int fd = connect_broker(host, port);
    if (fd < 0) {
        std::fprintf(stderr, "could not connect to broker %s:%d (is ./server running?)\n",
                     host.c_str(), port);
        return 1;
    }
    const std::string sub = "SUBSCRIBE " + topic + "\n";
    ::send(fd, sub.data(), sub.size(), 0);
    std::printf("consumer: connected to %s:%d, subscribed to '%s'\n", host.c_str(), port, topic.c_str());

    Consumer device(1024, policy);

    std::string buf;
    char chunk[4096];
    std::size_t n_events = 0;
    while (true) {
        const ssize_t r = ::recv(fd, chunk, sizeof(chunk), 0);
        if (r <= 0) { std::printf("consumer: broker closed connection\n"); break; }
        buf.append(chunk, static_cast<std::size_t>(r));

        std::size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            const std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (line.empty()) continue;

            MarketEvent event;
            try {
                event = Serializer::from_json(line);
            } catch (...) {
                continue;  // skip malformed
            }
            device.on_event(event);

            if (++n_events % static_cast<std::size_t>(print_every) == 0) {
                std::printf("n=%zu true_rho_s=%+.3f | ", n_events, event.true_rho_s);
                for (const auto& [name, rho] : device.dependence()) {
                    std::printf("%s=%+.3f ", name.c_str(), rho);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
        }
    }

    ::close(fd);
    return 0;
}
