#!/usr/bin/env bash
# Parallel full grid: every scenario x every DGP copula family, one process per job.
# Runs are independent (each writes its own results/multiseed_<scenario>[_<family>].csv)
# and use ~1 MB RSS each, so -P is bounded by cores, not memory.
# Deterministic: seeds are BASE..BASE+SEEDS-1 per job, so output matches the serial script.
set -euo pipefail
SEEDS="${1:-20}"
BASE="${2:-42}"
JOBS="${3:-$(sysctl -n hw.ncpu)}"
mkdir -p results

export SEEDS BASE

run_one() {
  # $1 = family, $2 = scenario
  COPULA="$1" ./multiseed "$2" "$SEEDS" "$BASE" > /dev/null || { echo "FAILED: $1/$2" >&2; exit 1; }
  echo "done: $1/$2"
}
export -f run_one

for family in gaussian clayton gumbel t; do
  for scenario in constant drift jump stochastic oscillate; do
    printf '%s %s\n' "$family" "$scenario"
  done
done | xargs -P "$JOBS" -L1 bash -c 'run_one "$0" "$1"'
