CXX = clang++

UWS = external/uWebSockets
USOCKETS = $(UWS)/uSockets/uSockets.a

OPENSSL = /opt/homebrew/opt/openssl
EIGEN = /opt/homebrew/opt/eigen
JSON = /opt/homebrew/opt/nlohmann-json

SERVER_SRC = \
src/app/main.cpp \
src/streamgenerator/datasource/synthetic_source.cpp \
src/streamgenerator/runtime/producer.cpp \
src/streamgenerator/protocol/serializer.cpp

SERVER_OUT = server

INCLUDES = \
-I$(UWS)/src \
-I$(UWS)/uSockets/src \
-I$(OPENSSL)/include \
-I$(EIGEN)/include/eigen3 \
-I$(JSON)/include

LIBS = \
-L$(OPENSSL)/lib \
-Wl,-rpath,$(OPENSSL)/lib \
$(USOCKETS) \
-lssl -lcrypto -lz

all: server

server:
	$(CXX) $(SERVER_SRC) -std=c++20 \
	$(INCLUDES) \
	$(LIBS) \
	-o $(SERVER_OUT)

# Standalone C++ consumer: connects to the broker over TCP, runs its own device
# + inference. Needs Eigen (Hermite) + nlohmann (deserialize); no uWebSockets.
CONSUMER_OUT = consumer

consumer:
	$(CXX) src/consumer/main.cpp src/streamgenerator/protocol/serializer.cpp -std=c++20 -O2 \
	-I$(EIGEN)/include/eigen3 -I$(JSON)/include \
	-o $(CONSUMER_OUT)

# NOTE: the offline experiment harness has its own Makefile (experiments/Makefile)
# and is intentionally NOT built here, so this repo has no dependency on it --
# experiments/ can be gitignored or split into a separate repo.

clean:
	rm -f server consumer

.PHONY: all server consumer clean