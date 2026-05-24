SyntheticSource source;
WebSocketServer server;

while (true) {
    auto event = source.next();

    auto json = Serializer::to_json(event);

    server.broadcast(json);

    std::this_thread::sleep_for(
        std::chrono::milliseconds(100)
    );
}