#!/usr/bin/env bash
# Builds every Chapter 6 simulation program into <folder>/build/. Needs g++ (or clang++), Eigen and the
# external/datasketches-cpp submodule. EIGEN=/usr on Linux, /opt/homebrew/opt/eigen on macOS.
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
EIGEN=${EIGEN:-/usr}
DS=../../external/datasketches-cpp
FLAGS="-std=c++20 -O2 -pthread -I$EIGEN/include/eigen3 -I$DS/common/include -I$DS/kll/include -I$DS/req/include"
for p in 6.2-fixed-forgetting/probe_fixed_ess common/probe_strategies 6.6-real-data/probe_real_practical \
         6.7-computational-cost/probe_budget 6.7-computational-cost/probe_margins; do
    d=$(dirname "$p"); n=$(basename "$p")
    mkdir -p "$d/build"
    (cd "$d" && $CXX "$n.cpp" $FLAGS -o "build/$n")
    echo "built $d/build/$n"
done
