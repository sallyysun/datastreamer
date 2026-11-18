#!/usr/bin/env bash
# Section 6.7 computational cost: matched memory budgets at n = 15,000 (validation seeds 1000-1039, test 42-141).
# SIM=1 reruns the simulations; otherwise the figures and tables are rebuilt from results/.
set -euo pipefail
cd "$(dirname "$0")"
KLL=8,16,32,64,128,256,512,1024,2048,4096,8192,16384
if [ "${SIM:-0}" = 1 ]; then
    for f in gaussian clayton gumbel t; do for s in constant drift jump; do
        BUDGET_N=15000 BUDGET_SCENARIOS=$s BUDGET_EST=xiao,hermite,mpl_dskll,mpl_exact BUDGET_KLL=$KLL \
            TAG=n15k_$s build/budget_benchmark sweep 100 40 $f
        BUDGET_N=15000 BUDGET_SCENARIOS=$s BUDGET_EST=mpl_kll BUDGET_KLL=$KLL \
            TAG=n15k_ownkll_$s build/budget_benchmark sweep 100 40 $f      # own KLL vs DataSketches
    done; done
    for r in 1 2 3 4 5; do
        BUDGET_N=15000 BUDGET_EST=xiao,hermite,mpl_dskll BUDGET_KLL=$KLL TAG=n15k_rep$r build/budget_benchmark time
        BUDGET_N=15000 BUDGET_EST=mpl_kll BUDGET_KLL=$KLL TAG=n15k_kll_rep$r build/budget_benchmark time
    done
    build/margin_ablation                                                  # margins_seeds (marginal-estimation diagnostic)
fi
python3 prepare_data.py                 # budget_{sweep,seeds}_n15k, budget_time_table_n15k
python3 select_configs.py               # budget_selected, budget_gain
python3 make_budget_config_tables.py    # tab:ch6-budget-configs, tab:budget-configurations-full
if [ "${REPORTS:-0}" = 1 ]; then
    python3 analyze_margins.py
fi
python3 plot_budget_storage.py          # ch6_budget_storage_n15k
python3 plot_budget_kll_n15k.py         # ch6_budget_kll_comparison
python3 plot_processing_time.py         # ch6_budget_processing_time
python3 plot_memory_curves.py           # ch6_budget_memory_curves_n15k
python3 plot_budget_gain_scenarios.py   # ch6_budget_accuracy_gain_scenarios
