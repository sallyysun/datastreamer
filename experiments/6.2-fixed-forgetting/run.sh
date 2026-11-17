#!/usr/bin/env bash
# Section 6.2 fixed forgetting. SIM=1 reruns the simulations (60 seeds x 4 families); otherwise the figures and tables
# are rebuilt from results/.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do COPULA=$f build/probe_fixed_ess 60; done
fi
python3 analyze_fixed_ess.py        # tab:optimal-ess-gauss, tab:optimal-ess-all-families
python3 analyze_jump_recovery.py    # tab:jump-recovery
python3 plot_constant_ess.py        # ch6_constant_ess_rmse, ch6_drift_ess_rmse
python3 plot_fixed_ess_biasvar.py   # ch6_drift_ess_bias_variance
python3 plot_jump_time.py           # ch6_jump_ess_rmse
python3 plot_jump_recovery.py       # ch6_jump_recovery_time
