#!/usr/bin/env bash
# Build inputs with inputs.py download <prices> and inputs.py build <prices> <workdir>/rerun.
# SIM=1 ./run.sh <workdir> compares strategies before regenerating figures.
# The thesis uses adjusted prices downloaded on 2026-11-15; later downloads may differ.
set -euo pipefail
cd "$(dirname "$0")"
S=$1
if [ "${SIM:-0}" = 1 ]; then
    for e in rmpl xiao hermite; do
        mkdir -p "$S/practical_$e/out"
        ls "$S/rerun/in" | xargs -P"$(getconf _NPROCESSORS_ONLN)" -I{} env EST=$e build/real_data_comparison "$S/rerun/in/{}" "$S/practical_$e/out/{}"
    done
    mkdir -p "$S/practical_roll2/out"   # + rolling-window benchmarks (rho_roll_<W>, rho_mplroll_<W>)
    ls "$S/rerun/in" | xargs -P"$(getconf _NPROCESSORS_ONLN)" -I{} env EST=rmpl ROLL=1 build/real_data_comparison "$S/rerun/in/{}" "$S/practical_roll2/out/{}"
fi
python3 analyse.py "$S"          # tables
python3 plots_loss.py "$S"       # loss figures
python3 plots_response.py "$S"   # response figures
python3 plots_examples.py "$S"   # ch6_practical_examples, ch6_practical_example_scatter (appendix)
