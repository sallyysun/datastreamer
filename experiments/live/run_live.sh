#!/usr/bin/env bash
# Score the standalone consumer over TCP and save one CSV per copula family.
# Usage: SEEDS=60 JOBS=8 ./run_live.sh
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-5}; JOBS=${JOBS:-$(nproc)}
FAMS=${FAMS:-"gaussian clayton gumbel t"}; SCENS=${SCENS:-"constant drift jump"}
SERVER=${SERVER:-../../server}; CONSUMER=${CONSUMER:-../../consumer}
study_defaults=$(python3 -c 'import json; c = json.load(open("../../study_config.json")); print(c["first_seed"], c["observations"])')
read -r FIRST_SEED OBSERVATIONS <<< "$study_defaults"
mkdir -p results/runs
one() {   # idx family scenario seed
    local i=$1 f=$2 sc=$3 sd=$4 p=$((20000 + 2 * $1))
    local out=results/runs/${f}_${sc}_${sd}.csv want=24
    [ "$sc" = jump ] && want=120
    [ -f "$out" ] && [ "$(wc -l < "$out")" -eq "$want" ] && return 0   # resume: this run is complete
    rm -f "$out"
    SEED=$sd N=$OBSERVATIONS RATE=0 WAIT_CONSUMERS=1 COPULA=$f DEPENDENCE=$sc PORT=$p BROKER_PORT=$((p + 1)) \
        "$SERVER" > results/runs/${f}_${sc}_${sd}.log 2>&1 &
    local spid=$!
    for _ in $(seq 100); do "$CONSUMER" --port $((p + 1)) --print 0 --score "$sc" --seed "$sd" --out "$out" \
        >> results/runs/${f}_${sc}_${sd}.log 2>&1 && break; sleep 0.1; done
    wait $spid || true
    [ -s "$out" ] || { echo "FAILED $f $sc $sd" >&2; return 1; }
}
export -f one; export SERVER CONSUMER OBSERVATIONS
i=0
for f in $FAMS; do for sc in $SCENS; do for ((s = 0; s < SEEDS; s++)); do
    echo "$i $f $sc $((FIRST_SEED + s))"; i=$((i + 1))
done; done; done | xargs -P "$JOBS" -L 1 bash -c 'one $0 $1 $2 $3'
for f in $FAMS; do
    sfx=$([ "$f" = gaussian ] && echo "" || echo "_$f")
    { echo "copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000"
      for sc in $SCENS; do for ((s = 0; s < SEEDS; s++)); do cat "results/runs/${f}_${sc}_$((FIRST_SEED + s)).csv"; done; done
    } > "results/live_seeds$sfx.csv"
done
echo "wrote results/live_seeds[_family].csv ($(ls results/runs/*.csv | wc -l) runs)"
