#!/usr/bin/env bash
# Section 6.2 fixed forgetting. SIM=1 reruns the simulations (60 seeds x 4 families); otherwise the figures and tables
# are rebuilt from results/.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do COPULA=$f build/fixed_ess_sweep; done
fi
python3 analyse.py   # tables
python3 plots.py     # figures
