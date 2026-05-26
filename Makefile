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

TEST_SRC = src/tests/test_stream.cpp

SERVER_OUT = server
TEST_OUT = test_stream

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

test:
	$(CXX) $(TEST_SRC) -std=c++20 \
	$(INCLUDES) \
	-o $(TEST_OUT)

clean:
	rm -f server test_stream