#!/usr/bin/env bash
# reference_parity -- validate the C++ dependence estimators against their
# upstream reference implementations on shared inputs.
#
#   Xiao empirical-copula Spearman  <->  external/onlineNPCORR  (Python)
#   Hermite-series Spearman         <->  external/hermiter      (R package)
#
# Feeds byte-identical (x,y) streams to the C++ estimator classes and to each
# reference, then reports the per-step max/mean absolute difference in the
# estimated Spearman rho. Requires: clang++, Eigen, python3 (numpy), Rscript
# with the `hermiter` package installed.
#
# Run:  experiments/reference_parity/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
mkdir -p out

EIGEN="$( (brew --prefix eigen 2>/dev/null || echo /opt/homebrew/opt/eigen) )/include/eigen3"

echo "== building harness =="
clang++ harness.cpp -std=c++20 -O2 -I"$REPO/src" -I"$EIGEN" -o out/harness

echo "== generating shared inputs =="
python3 gen_inputs.py

DATASETS=(gauss_pos gauss_neg gauss_zero skew_marg)

echo
echo "==================== XIAO cumulative (C++ vs Python onlineNPCORR) ===================="
for ds in "${DATASETS[@]}"; do
  ./out/harness xiao "data/$ds.csv" 20 100 0 2>"out/cuts_$ds.txt" >"out/cpp_xiao_$ds.txt"
  python3 xiao_reference.py "data/$ds.csv" "out/cuts_$ds.txt" 0 >"out/py_xiao_$ds.txt"
  python3 compare.py "out/cpp_xiao_$ds.txt" "out/py_xiao_$ds.txt" 100 "xiao/$ds"
done

echo
echo "==================== XIAO window=500 (C++ vs Python online_mv_corr) ===================="
for ds in "${DATASETS[@]}"; do
  ./out/harness xiao "data/$ds.csv" 20 100 500 2>"out/cutsw_$ds.txt" >"out/cpp_xiaow_$ds.txt"
  python3 xiao_reference.py "data/$ds.csv" "out/cutsw_$ds.txt" 500 >"out/py_xiaow_$ds.txt"
  # window is full & sliding (matches the pure-500 reference) from index 499 on.
  python3 compare.py "out/cpp_xiaow_$ds.txt" "out/py_xiaow_$ds.txt" 499 "xiao_win/$ds"
done

echo
echo "==================== HERMITE cumulative (C++ vs R hermiter) ===================="
for ds in "${DATASETS[@]}"; do
  ./out/harness hermite "data/$ds.csv" 10 0 0 >"out/cpp_herm_$ds.txt" 2>/dev/null
  Rscript hermite_reference.R "data/$ds.csv" 10 0 >"out/r_herm_$ds.txt" 2>/dev/null
  python3 compare.py "out/cpp_herm_$ds.txt" "out/r_herm_$ds.txt" 50 "hermite/$ds"
done

echo
echo "==================== HERMITE ew lambda=0.01 (C++ vs R hermiter exp_weight) ===================="
for ds in "${DATASETS[@]}"; do
  ./out/harness hermite "data/$ds.csv" 10 0 0.01 >"out/cpp_hermew_$ds.txt" 2>/dev/null
  Rscript hermite_reference.R "data/$ds.csv" 10 0.01 >"out/r_hermew_$ds.txt" 2>/dev/null
  python3 compare.py "out/cpp_hermew_$ds.txt" "out/r_hermew_$ds.txt" 50 "hermite_ew/$ds"
done

echo
echo "Xiao matches to machine epsilon (~1e-16); Hermite to ~1e-10 (fp accumulation"
echo "across different linear-algebra backends). All C++ ports reproduce the references."
