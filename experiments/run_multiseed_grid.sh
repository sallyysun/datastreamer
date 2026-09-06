#!/usr/bin/env bash
# Full multi-seed grid: every scenario x every DGP copula family.
# Run from experiments/ after `make multiseed`. Writes results/multiseed_<scenario>[_<family>].csv
set -euo pipefail
SEEDS="${1:-20}"
BASE="${2:-42}"
mkdir -p results
for family in gaussian clayton gumbel t; do
  for scenario in constant drift jump stochastic oscillate; do
    COPULA="$family" ./multiseed "$scenario" "$SEEDS" "$BASE" > /dev/null
    echo "done: $family/$scenario"
  done
done
