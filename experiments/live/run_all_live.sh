#!/usr/bin/env bash
# Run jobs.txt over TCP and resume from jobs marked done.
# Use compare_all_live.py to compare the resulting CSVs with in-process runs.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-$(nproc)}
export LIVE=1 LIVE_SERVER=$(cd ../.. && pwd)/server
job() {   # line
    IFS='|' read -r sec envs prog seeds <<< "$1"
    sec=$(echo $sec); envs=$(echo $envs); prog=$(echo $prog); seeds=$(echo $seeds)
    local log="logs/$(echo "$sec $envs ${prog##*/}" | tr ' =,/' '____').log"
    grep -qs "^done" "$log" && return 0
    mkdir -p "$sec/results" logs
    ( cd "$sec" && env $envs $prog ${seeds:+"$seeds"} ) > "$log" 2>&1 && echo "done" >> "$log" || echo "FAILED: $sec $envs" >&2
}
export -f job
mkdir -p logs
grep -v '^#' jobs.txt | grep -v '^$' | xargs -P "$JOBS" -d '\n' -I{} bash -c 'job "$@"' _ {}
echo "finished: $(grep -ls '^done' logs/*.log | wc -l) of $(grep -vc '^#' jobs.txt) jobs"
