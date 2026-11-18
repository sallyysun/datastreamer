#!/usr/bin/env bash
# Section 6.3 adaptive forgetting (fixed vs VFF vs MFF). SIM=1 reruns the simulations with the shared runner
# (../common/build/strategy_experiments); otherwise the figures are rebuilt from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/strategy_experiments
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do
        COPULA=$f $P                                                         # strategies_seeds, strategies_jumppath
        COPULA=$f TAG=vffsens VFF_RATES=0.005,0.01,0.02,0.05,0.075,0.1,0.2 $P # VFF rate sensitivity (appendix)
    done
    PATHS=1 TAG=est $P     # strategies_paths_est (trajectories)
    PATHS=1 $P             # strategies_paths
    WEIGHTS=1 $P           # strategies_weights (MFF weight paths, delta = 1e-4 and 0)
fi
python3 analyze_strategies.py
python3 analyze_vffsens.py
METRIC=rmse python3 plot_strategy_allfixed.py   # ch6_strategy_rmse_bars_allfixed
python3 plot_strategies.py                      # ch6_rmse_trajectories, ch6_jump_trajectories
python3 plot_mff_weights.py                     # ch6_mff_weights_*, ch6_mff_fixedshare_jump_*
python3 plot_vffsens.py                         # ch6_vffsens_rmse, ch6_vffsens_h2
