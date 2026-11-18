#!/usr/bin/env bash
# Builds the Chapter 6 simulation programs (experiments/simulation/) into each
# section's build/ folder, and the two consistency tests into common/build/. Needs g++ (or clang++), Eigen,
# nlohmann-json (the live mode's tick parser) and the external/datasketches-cpp submodule. EIGEN=/usr JSON=/usr on Linux,
# /opt/homebrew/opt/eigen and /opt/homebrew/opt/nlohmann-json on macOS.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-g++}
EIGEN=${EIGEN:-/usr}
JSON=${JSON:-/usr}
DS=external/datasketches-cpp
FLAGS="-std=c++20 -O2 -pthread -I$EIGEN/include/eigen3 -I$JSON/include -I$DS/common/include -I$DS/kll/include"
build() {   # source, output
    mkdir -p "$(dirname "$2")"
    $CXX "$1" src/protocol/serializer.cpp $FLAGS -o "$2"
    echo "built $2"
}
build experiments/simulation/probe_fixed_ess.cpp             experiments/6.2-fixed-forgetting/build/probe_fixed_ess
build experiments/simulation/scenarios.cpp  experiments/common/build/scenarios
build experiments/simulation/probe_real_practical.cpp        experiments/6.6-real-data/build/probe_real_practical
build experiments/simulation/probe_budget.cpp                experiments/6.7-computational-cost/build/probe_budget
build experiments/simulation/probe_margins.cpp               experiments/6.7-computational-cost/build/probe_margins
build experiments/simulation/test/device_vs_probe.cpp        experiments/common/build/device_vs_probe
build experiments/simulation/test/lab_vs_device.cpp          experiments/common/build/lab_vs_device
