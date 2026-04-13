#!/usr/bin/env bash

# Loads the necessary HPC modules and executes whatever is passed to it.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

module load numactl CUDA
exec -- "$@"
