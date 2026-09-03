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

clean:
	rm -f experiment calibrate_arl0

.PHONY: experiment calibrate_arl0 clean
