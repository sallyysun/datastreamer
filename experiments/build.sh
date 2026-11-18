#!/usr/bin/env bash
# Build the Chapter 6 runners and consistency checks. Requires C++20, Eigen, nlohmann-json and DataSketches.
# Set EIGEN and JSON to the dependency prefixes.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-g++}
EIGEN=${EIGEN:-/usr}
JSON=${JSON:-/usr}
DS=external/datasketches-cpp
make -s experiments/common/build/study_config.hpp
FLAGS="-Iexperiments/common/build -std=c++20 -O2 -pthread -I$EIGEN/include/eigen3 -I$JSON/include -I$DS/common/include -I$DS/kll/include"
build() {   # source, output, additional sources
    local source=$1 output=$2
    shift 2
    mkdir -p "$(dirname "$output")"
    $CXX "$source" "$@" $FLAGS -o "$output"
}
build experiments/6.2-fixed-forgetting/fixed_ess_sweep.cpp experiments/6.2-fixed-forgetting/build/fixed_ess_sweep
build experiments/simulation/experiment.cpp                experiments/common/build/strategy_experiments \
    experiments/simulation/simulation.cpp \
    experiments/simulation/simulation_reports.cpp \
    experiments/simulation/validate.cpp \
    experiments/simulation/family.cpp \
    experiments/simulation/regret.cpp
build experiments/6.6-real-data/real_data_comparison.cpp   experiments/6.6-real-data/build/real_data_comparison
build experiments/6.7-computational-cost/budget_benchmark.cpp  experiments/6.7-computational-cost/build/budget_benchmark
build experiments/6.7-computational-cost/margin_comparison.cpp experiments/6.7-computational-cost/build/margin_comparison
build experiments/simulation/test/device_vs_probe.cpp      experiments/common/build/device_vs_probe
build experiments/simulation/test/lab_vs_device.cpp        experiments/common/build/lab_vs_device
