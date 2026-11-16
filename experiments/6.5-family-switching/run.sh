#!/usr/bin/env bash
# Section 6.5 family switching (Gaussian-only vs family-augmented MFF). SIM=1 reruns the simulations with the
# shared probe (../common/build/probe_strategies); otherwise from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/probe_strategies
if [ "${SIM:-0}" = 1 ]; then
    FAMSWITCH=1 $P 60                                                    # famsw_{seeds,ident,path}_<regime>
    for e in rmpl xiao hermite; do FAMSWITCH=1 FS_H2=1 FS_EST=$e $P 60; done   # famsw_seeds_h2_<est>_<regime>
fi
python3 famsw_rmpl.py               # famsw_rmpl_dl, ch6_famswitch_rmpl_weights
python3 famsw_accuracy.py           # ch6_famswitch_accuracy
python3 plot_famsw_initial_id.py    # ch6_famswitch_initial_id
python3 plot_famsw_adaptation.py    # ch6_famswitch_adaptation
