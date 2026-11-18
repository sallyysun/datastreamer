#!/usr/bin/env bash
# Build inputs with download_prices.py <prices> and build_inputs.py <prices> <workdir>/rerun.
# SIM=1 ./run.sh <workdir> compares strategies before regenerating figures.
# The thesis uses adjusted prices downloaded on 2026-11-15; later downloads may differ.
set -euo pipefail
cd "$(dirname "$0")"
S=$1
if [ "${SIM:-0}" = 1 ]; then
    for e in rmpl xiao hermite; do
        mkdir -p "$S/practical_$e/out"
        ls "$S/rerun/in" | xargs -P"$(nproc)" -I{} env EST=$e build/real_data_comparison "$S/rerun/in/{}" "$S/practical_$e/out/{}"
    done
    mkdir -p "$S/practical_roll2/out"   # + rolling-window benchmarks (rho_roll_<W>, rho_mplroll_<W>)
    ls "$S/rerun/in" | xargs -P"$(nproc)" -I{} env EST=rmpl ROLL=1 build/real_data_comparison "$S/rerun/in/{}" "$S/practical_roll2/out/{}"
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
