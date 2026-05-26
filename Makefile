CXX = clang++

UWS = external/uWebSockets
USOCKETS = $(UWS)/uSockets/uSockets.a

SRC = $(shell find src -name "*.cpp")
OUT = server

OPENSSL = /opt/homebrew/opt/openssl
EIGEN = /opt/homebrew/opt/eigen
JSON = /opt/homebrew/opt/nlohmann-json

all:
	$(CXX) $(SRC) -std=c++20 \
	-I$(UWS)/src \
	-I$(UWS)/uSockets/src \
	-I$(OPENSSL)/include \
	-I$(EIGEN)/include/eigen3 \
	-I$(JSON)/include \
	-L$(OPENSSL)/lib \
	-Wl,-rpath,$(OPENSSL)/lib \
	$(USOCKETS) \
	-lssl -lcrypto -lz \
	-o $(OUT)

clean:
	rm -f server