#!/usr/bin/env bash

# Run this on the cluster. Runs all other benchmarks.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

scripts/srun.sh --cpu scripts/with_modules.sh scripts/bench_hpc_seq.sh
scripts/srun.sh --cpu --cpus=64 scripts/with_modules.sh scripts/bench_hpc_omp.sh
scripts/srun.sh --gpu scripts/with_modules.sh scripts/bench_hpc_cuda.sh
