#!/usr/bin/env bash
# Section 6.4 copula density: log vs squared loss across families, density accuracy, MFF regret.
# SIM=1 reruns the simulations with the shared runner (../common/build/strategy_experiments); otherwise from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/strategy_experiments
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do
        COPULA=$f SCORESFAM=1 TAG=sfeval $P     # scoresfam_seeds_sfeval (evaluation seeds 42-101)
        COPULA=$f REGRETFAM=1 $P                # regretfam_seeds
        COPULA=$f MIXDENS=1 TAG=mix $P          # strategies_seeds_mix (MFF density = mixture; density bars)
    done
    REGRET=1 $P                                 # regret_seeds, regret_paths
fi
python3 analyze_scoresfam.py                       # scoresfam_cells
if [ "${REPORTS:-0}" = 1 ]; then
    python3 analyze_regret.py all
fi
python3 plot_scoresfam.py rmpl xiao hermite        # ch6_scoresfam_heatmap_*
python3 ../common/strategy_allfixed.py . density  # ch6_strategy_density_bars_allfixed
python3 plot_mff_regret_summary.py                 # ch6_mff_regret_summary, ch6_mff_regret_jump
