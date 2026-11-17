#!/usr/bin/env bash
# Section 6.6 real data: 32 daily-return pairs (2006 onwards), five fixed RMPL / Counting Matrix / Hermite arms, VFF and
# MFF. Raw prices are not redistributed. S = a work directory outside the repository.
#   1. python3 download_prices.py <prices>          Yahoo adjusted closes + ^VIX (a fresh download is not byte-identical:
#                                                   Yahoo revises adjusted history; the thesis data are from 2026-11-15)
#   2. python3 build_inputs.py <prices> $S/rerun    32 pairs, EWMA(0.94)-standardised log returns -> $S/rerun/in, vix.csv
#   3. SIM=1 ./run.sh $S                            probe runs (below), then the figures
#   ./run.sh $S                                     figures only, from existing probe outputs
set -euo pipefail
cd "$(dirname "$0")"
S=$1
if [ "${SIM:-0}" = 1 ]; then
    for e in rmpl xiao hermite; do
        mkdir -p "$S/practical_$e/out"
        ls "$S/rerun/in" | xargs -P"$(nproc)" -I{} env EST=$e build/probe_real_practical "$S/rerun/in/{}" "$S/practical_$e/out/{}"
    done
    mkdir -p "$S/practical_roll2/out"   # + rolling-window benchmarks (rho_roll_<W>, rho_mplroll_<W>)
    ls "$S/rerun/in" | xargs -P"$(nproc)" -I{} env EST=rmpl ROLL=1 build/probe_real_practical "$S/rerun/in/{}" "$S/practical_roll2/out/{}"
fi
python3 analyze_switching.py "$S"                     # switching_per_pair (read by the regret figure)
python3 analyze_hedge_change.py "$S"                  # hedge_change_results (read by the hedge figure)
python3 plot_final_regret.py "$S"                     # ch6_practical_final_regret
python3 plot_change_days_levels.py "$S"               # ch6_practical_final_change_days
python3 plot_main_weights_ess.py "$S"                 # ch6_practical_weights_ess
python3 plot_main_response.py "$S"                    # ch6_practical_response
python3 plot_appendix_vix.py "$S" "$S/rerun/vix.csv"  # ch6_practical_appendix_vix
python3 plot_appendix_hedge_mini.py "$S"              # ch6_practical_hedge_equal_mini
python3 plot_strength_vs_advantage.py "$S"            # ch6_practical_strength_vs_advantage
