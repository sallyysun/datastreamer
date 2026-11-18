#!/usr/bin/env bash
# Section 6.5 family switching (Gaussian-only vs family-augmented MFF). SIM=1 reruns the simulations with the
# shared runner (../common/build/strategy_experiments); otherwise from results/.
set -euo pipefail
cd "$(dirname "$0")"
P=../common/build/strategy_experiments
if [ "${SIM:-0}" = 1 ]; then
    for g in gaussian t clayton gumbel gauss2t gauss2clayton clayton2gumbel; do
        FAMSWITCH=1 FS_GENS=$g FS_TAG=$g $P                             # famsw_{seeds,ident,path}_<generator>
    done
    for g in gauss2t gauss2clayton clayton2gumbel; do for e in rmpl xiao hermite; do
        FAMSWITCH=1 FS_H2=1 FS_EST=$e FS_GENS=$g FS_TAG=h2_${e}_$g $P   # famsw_seeds_h2_<est>_<switch>
    done; done
fi
python3 analyze_family_switching.py all
python3 plot_family_switching.py all
