#!/usr/bin/env bash
# Every Chapter 6.2-6.5 simulation over the live pipeline: each job in jobs.txt runs its probe with LIVE=1 (one seeded
# server per stream, observations over TCP; experiments/simulation/live_source.hpp) inside live/<section>/, writing
# live/<section>/results/ under the same file names as the in-process run. Resumable: a job whose log ends in "done"
# is skipped. Then compare_all_live.py compares every file with <section>/results/.
# Usage: JOBS=4 ./run_all_live.sh     (needs ../../server and ../build.sh)
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-$(nproc)}
export LIVE=1 LIVE_SERVER=$(cd ../.. && pwd)/server
job() {   # line
    IFS='|' read -r sec envs prog seeds <<< "$1"
    sec=$(echo $sec); envs=$(echo $envs); prog=$(echo $prog); seeds=$(echo $seeds)
    local log="logs/$(echo "$sec $envs" | tr ' =,/' '____').log"
    grep -qs "^done" "$log" && return 0
    mkdir -p "$sec/results" logs
    ( cd "$sec" && env $envs "$prog" "$seeds" ) > "$log" 2>&1 && echo "done" >> "$log" || echo "FAILED: $sec $envs" >&2
}
export -f job
mkdir -p logs
grep -v '^#' jobs.txt | grep -v '^$' | xargs -P "$JOBS" -d '\n' -I{} bash -c 'job "$@"' _ {}
echo "finished: $(grep -ls '^done' logs/*.log | wc -l) of $(grep -vc '^#' jobs.txt) jobs"
