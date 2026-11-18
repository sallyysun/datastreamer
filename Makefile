CXX ?= clang++
PYTHON ?= python3
STUDY_HEADER = experiments/common/build/study_config.hpp
STUDY_INC = -I$(dir $(STUDY_HEADER))

UWS = external/uWebSockets
USOCKETS = $(UWS)/uSockets/uSockets.a

OPENSSL ?= /opt/homebrew/opt/openssl
EIGEN ?= /opt/homebrew/opt/eigen
JSON ?= /opt/homebrew/opt/nlohmann-json
DS = external/datasketches-cpp
DS_INC = -I$(DS)/common/include -I$(DS)/kll/include -I$(DS)/req/include

SERVER_SRC = src/streamgenerator/main.cpp

SERVER_OUT = server

INCLUDES = \
-I$(UWS)/src \
-I$(UWS)/uSockets/src \
-I$(OPENSSL)/include \
-I$(EIGEN)/include/eigen3 \
-I$(JSON)/include \
$(DS_INC) $(STUDY_INC)

LIBS = \
-L$(OPENSSL)/lib \
-Wl,-rpath,$(OPENSSL)/lib \
$(USOCKETS) \
-lssl -lcrypto -lz

all: server

$(STUDY_HEADER): study_config.json experiments/common/write_study_config.py
	@$(PYTHON) experiments/common/write_study_config.py $< $@

server: $(STUDY_HEADER)
	$(CXX) $(SERVER_SRC) -std=c++20 \
	$(INCLUDES) \
	$(LIBS) \
	-o $(SERVER_OUT)

# Standalone C++ consumer: connects to the broker over TCP, runs its own device
# + inference (the thesis VFF/MFF strategies). Needs Eigen (Hermite), nlohmann (deserialize) and the
# datasketches submodule (RMPL's KLL margin); no uWebSockets.
CONSUMER_OUT = consumer

consumer: $(STUDY_HEADER)
	$(CXX) src/consumer/main.cpp -std=c++20 -O2 \
	-I$(EIGEN)/include/eigen3 -I$(JSON)/include $(DS_INC) $(STUDY_INC) -pthread \
	-o $(CONSUMER_OUT)

# The Chapter 6 experiments are built from their own folders (experiments/README.md).

clean:
	rm -f server consumer

.PHONY: all server consumer clean
