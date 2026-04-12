#!/usr/bin/env bash

# Compiles and runs the current codebase using the given k=v flags.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

export OMP_PLACES=cores
export OMP_PROC_BIND=close

if [ -v SLURM_CPUS_PER_TASK ]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

cmd=$(scripts/make.sh "$@")
"$cmd" "$@"
