#!/usr/bin/env bash
# Section 6.4 copula density: log vs squared loss across families, density accuracy, MFF regret.
# SIM=1 reruns the simulations with the shared runner (../common/build/strategy_experiments); otherwise from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/strategy_experiments
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do
        COPULA=$f TAG=sfeval $P score-family # scoresfam_seeds_sfeval (evaluation seeds 42-101)
        COPULA=$f $P regret-family           # regretfam_seeds
        COPULA=$f TAG=mix $P mixture         # strategies_seeds_mix (MFF density = mixture; density bars)
    done
    $P regret                                # regret_seeds, regret_paths
fi
python3 analyse.py   # tables
python3 plots.py     # figures
