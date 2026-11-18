#!/usr/bin/env bash
# Quick reproducibility check: builds the simulation programs, runs every mode on a short run (2 seeds, a short
# budget sweep, a synthetic real-data pair), and compares each output's SHA-256 with experiments/check_reference.txt.
# Linux/libstdc++ only (std::*_distribution draws differ on macOS).
#   experiments/check.sh            build, run, compare (exit 1 on any difference)
#   experiments/check.sh --update   rewrite the reference after an intended change
#   NOBUILD=1                       skip the build;  KEEP=<dir> keeps the outputs there
set -euo pipefail
cd "$(dirname "$0")"
E=$PWD
[ "${NOBUILD:-0}" = 1 ] || ./build.sh > /dev/null
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/results"
cd "$D"
exec 3>&1 > "$D/run.log" 2>&1
trap 'exec 1>&3 2>&3; echo "a run failed; last lines of its output:"; tail -20 "$D/run.log"' ERR
S=$E/common/build/strategy_experiments
$S 2
COPULA=clayton $S 2
MIXDENS=1 TAG=mix $S 2
WEIGHTS=1 $S 2
REGRET=1 $S 2
COPULA=t REGRETFAM=1 $S 2
FAMSWITCH=1 FS_H2=1 FS_GENS=gauss2t $S 2
FAMSWITCH=1 FS_H2=1 FS_EST=hermite FS_GENS=clayton2gumbel $S 2
COPULA=gumbel SCORESFAM=1 TAG=sfeval $S 2
PATHS=1 $S 2
TAG=vffsens VFF_RATES=0.005,0.2 $S 2
VALIDATE=1 $S 1 > results/validate_strategies.txt
F=$E/6.2-fixed-forgetting/build/fixed_ess_sweep
$F 2
COPULA=gumbel $F 2
B=$E/6.7-computational-cost/build/budget_benchmark
BUDGET_N=1500 BUDGET_KLL=8,64,512 TAG=small $B sweep 2 1
BUDGET_N=1500 BUDGET_KLL=8,64,512 BUDGET_EST=mpl_kll TAG=own $B sweep 2 1 clayton
python3 - <<'PY'
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 1500
z = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1, 0.6], [0.6, 1]]).T
pd.DataFrame({"date": pd.bdate_range("2006-01-03", periods=n).strftime("%Y-%m-%d"), "x": z[:, 0], "y": z[:, 1]}).to_csv("in.csv", index=False, float_format="%.10g")
PY
R=$E/6.6-real-data/build/real_data_comparison
for e in rmpl xiao hermite; do
    EST=$e $R in.csv results/real_$e.csv
    EST=$e ROLL=1 $R in.csv results/realroll_$e.csv
done
(cd results && sha256sum * | sort -k2) > sums.txt
exec 1>&3 2>&3
[ -z "${KEEP:-}" ] || { mkdir -p "$KEEP" && cp results/* "$KEEP"/; }
if [ "${1:-}" = --update ]; then
    cp sums.txt "$E/check_reference.txt"
    echo "reference updated: $(wc -l < sums.txt) outputs"
    exit 0
fi
if diff <(awk '{print $2, $1}' "$E/check_reference.txt") <(awk '{print $2, $1}' sums.txt) > diff.txt; then
    echo "all $(wc -l < sums.txt) outputs match the reference"
else
    echo "outputs that differ from the reference:"
    grep '^[<>]' diff.txt | awk '{print "  " $2}' | sort -u
    exit 1
fi
