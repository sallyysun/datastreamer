# Offline estimator evaluation harness -- self-contained.
#
# Depends only on Eigen and the estimator headers under ../src (pulled in by the
# relative includes in main.cpp). It does NOT link the streaming server, so this
# folder can be gitignored or split into its own repo. Run from THIS directory:
#
#   make && ./experiment jump        # writes results/experiment_jump.csv
#   for s in constant drift jump stochastic oscillate; do ./experiment $s; done

CXX = clang++
EIGEN = /opt/homebrew/opt/eigen

experiment:
	$(CXX) main.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o experiment

calibrate_arl0:
	$(CXX) calibrate_arl0.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o calibrate_arl0

multiseed:
	$(CXX) multiseed.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o multiseed

async_memory:
	$(CXX) async_memory.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o async_memory

clean:
	rm -f experiment calibrate_arl0 multiseed async_memory

.PHONY: experiment calibrate_arl0 multiseed async_memory clean
