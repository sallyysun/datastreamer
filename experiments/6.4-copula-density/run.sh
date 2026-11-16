#!/usr/bin/env bash
# Section 6.4 copula density: log vs squared loss across families, density accuracy, MFF regret.
# SIM=1 reruns the simulations with the shared probe (../common/build/probe_strategies); otherwise from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/probe_strategies
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do
        COPULA=$f SCORESFAM=1 TAG=sfeval $P 60     # scoresfam_seeds_sfeval (evaluation seeds 42-101)
        COPULA=$f REGRETFAM=1 $P 60                # regretfam_seeds
        COPULA=$f MIXDENS=1 TAG=mix $P 60          # strategies_seeds_mix (MFF density = mixture; density bars)
    done
    REGRET=1 $P 60                                 # regret_seeds, regret_paths
fi
python3 analyze_scoresfam.py                       # scoresfam_cells
python3 analyze_regret.py                          # tab:mff-regret-bounds
python3 analyze_regretfam.py                       # tab:app-mff-regret-families
python3 plot_scoresfam.py rmpl xiao hermite        # ch6_scoresfam_heatmap_*
METRIC=density python3 plot_strategy_allfixed.py   # ch6_strategy_density_bars_allfixed
python3 plot_mff_regret_summary.py                 # ch6_mff_regret_summary, ch6_mff_regret_jump
