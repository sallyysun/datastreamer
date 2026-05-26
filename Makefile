CXX = clang++

UWS = external/uWebSockets
USOCKETS = $(UWS)/uSockets/uSockets.a

APP = src/app/main.cpp
OUT = server

OPENSSL = /opt/homebrew/opt/openssl

all:
	$(CXX) $(APP) -std=c++20 \
	-I$(UWS)/src \
	-I$(UWS)/uSockets/src \
	-I$(OPENSSL)/include \
	-L$(OPENSSL)/lib \
	-Wl,-rpath,$(OPENSSL)/lib \
	$(USOCKETS) \
	-lssl -lcrypto -lz \
	-o $(OUT)

clean:
	rm -f server