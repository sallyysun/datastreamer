#!/usr/bin/env bash
# Chapter 3 background figures (needs scipy).
set -euo pipefail
cd "$(dirname "$0")"
python3 plot_ch3_parametric.py          # ch3_parametric
python3 plot_ch3_empirical_nobeta.py    # ch3_empirical
