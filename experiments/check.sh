#!/usr/bin/env bash
# Quick reproducibility check: builds the simulation programs, runs every mode on a short run (2 seeds, a short
# budget sweep, a synthetic real-data pair), and compares each output's SHA-256 with experiments/check_reference.txt.
# Then the two consistency tests: the live device must reproduce the runner's rows exactly (device_vs_probe), and the
# browser lab's MFF and VFF must equal the device's (lab_vs_device). GitHub runs this on every push (.github/workflows).
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
$S strategies 2
COPULA=clayton $S strategies 2
TAG=mix $S mixture 2
$S weights 2
$S regret 2
COPULA=t $S regret-family 2
FS_H2=1 FS_GENS=gauss2t $S family-switching 2
FS_H2=1 FS_EST=hermite FS_GENS=clayton2gumbel $S family-switching 2
COPULA=gumbel TAG=sfeval $S score-family 2
PATHS=1 $S strategies 2
TAG=vffsens VFF_RATES=0.005,0.2 $S strategies 2
$S validate 1 > results/validate_strategies.txt
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
T=$E/common/build
for sc in constant drift jump; do for seed in 42 43; do $T/device_vs_probe gaussian $sc $seed; done; done > device_rows.csv
$T/lab_vs_device gaussian jump 42 > lab_vs_device.txt
exec 1>&3 2>&3
tests_ok=1
if ! python3 - results/strategies_seeds_mix.csv device_rows.csv <<'PY'
import sys
import pandas as pd
runner = pd.read_csv(sys.argv[1])
device = pd.read_csv(sys.argv[2], header=None, names=runner.columns)
key = ["copula", "scenario", "estimator", "strategy", "seed", "window"]
both = device.merge(runner, on=key, suffixes=("_device", "_runner"))
worst = max((both[c + "_device"] - both[c + "_runner"]).abs().max() for c in ["rmse", "excess_x1000", "h2_x1000"])
ok = len(device) > 0 and len(both) == len(device) and worst == 0
print(f"device_vs_probe: {len(both)} of {len(device)} device rows match the runner, max difference {worst:g}")
sys.exit(0 if ok else 1)
PY
then tests_ok=0; fi
if ! python3 - lab_vs_device.txt <<'PY'
import re, sys
lines = open(sys.argv[1]).read().split("\n")
diffs = [float(v) for line in lines for v in re.findall(r"\| ([-+0-9.eE]+|nan|inf)", line)]
ok = len(diffs) == 6 and all(d == 0 for d in diffs)
print("lab_vs_device: lab MFF and VFF equal the device's for all three estimators" if ok
      else "lab_vs_device: the lab and the device disagree:\n" + "\n".join(lines))
sys.exit(0 if ok else 1)
PY
then tests_ok=0; fi
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
[ "$tests_ok" = 1 ] || exit 1
