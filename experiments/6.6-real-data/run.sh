#!/usr/bin/env bash
# Section 6.6 on the thesis inputs in data/ (Yahoo adjusted closes downloaded 2026-11-15, last day 2026-11-11).
#   SIM=1 ./run.sh [workdir]   run the estimators on every pair, then the tables and figures
#   ./run.sh <workdir>         tables and figures from outputs already in <workdir> (e.g. the data-66-thesis archives)
# workdir defaults to results/work; it gets rerun/ (a copy of data/) if it has none.
# New inputs: inputs.py download <prices> and inputs.py build <prices> <workdir>/rerun (later downloads differ).
set -euo pipefail
cd "$(dirname "$0")"
S=${1:-results/work}
if [ ! -d "$S/rerun" ]; then
    mkdir -p "$S/rerun"
    cp -R data/in data/vix.csv "$S/rerun/"
fi
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
