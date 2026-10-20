#!/bin/bash
# Bit-exact gate for the probes that build estimators outside registry.hpp (trajectory_manifest.hpp).
#   ./check_probe_trajectories.sh before/ ; <change> ; ./check_probe_trajectories.sh after/ ; diff -r before after
# Small seed counts where a probe takes them. EIGEN=<prefix> is passed to make (default: Makefile's).
set -euo pipefail
cd "$(dirname "$0")"
[ $# -eq 1 ] || { echo "usage: $0 <out_dir>" >&2; exit 1; }
mkdir -p "$1"
out=$(cd "$1" && pwd)
here=$(pwd)

probes=(probe_memory probe_adaptive probe_vff_norm async_memory async_noise tail_validate tail_change tail_mixture tail_soft)
for p in "${probes[@]}"; do make -s -B "$p" ${EIGEN:+EIGEN=$EIGEN}; done

# Probes overwrite their results/ CSVs; run in a scratch dir so the real ones survive.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/results"

run() {
    local tag=$1; shift
    local t0=$SECONDS
    (cd "$work" && TRAJ_MANIFEST="$out/$tag.txt" "$here/$1" "${@:2}" > /dev/null)
    echo "$tag: $((SECONDS - t0))s, $(wc -l < "$out/$tag.txt") series"
}

run probe_memory   probe_memory sweep
run probe_memory_ess probe_memory ess
run probe_adaptive probe_adaptive
run probe_vff_norm probe_vff_norm
run async_memory   async_memory drift 2
run async_noise    async_noise 2
run tail_validate  tail_validate 0.6 2
run tail_change    tail_change 2
run tail_mixture   tail_mixture
run tail_soft      tail_soft 2
echo "manifests in $out ($SECONDS s total)"
