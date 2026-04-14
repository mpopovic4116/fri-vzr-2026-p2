#!/usr/bin/env bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

run_bench() {
    cmd=$(scripts/make.sh "${@:2}")
    for _i in $(seq -- "$1"); do
        "$cmd" "${@:2}"
    done
}

run_bench_for_config() {
    for size in 256 512 1024 2048 4096; do
        run_bench 5 "$@" size_w="$size" size_h="$size"
    done
}

args=("$@")
cuda_bench() {
    run_bench_for_config "${args[@]}" impl=cuda cuda_arch=v100s "$@"
}

for precision in 64 32; do
    for kernel in default shared fused; do
        cuda_bench precision="$precision" kernel="$kernel"
    done
done

for w in 32 64 128 256 512 1024; do
    for h in 32 64 128 256 512 1024; do
        cuda_bench precision=32 kernel=fused threads_x="$w" threads_y="$h"
    done
done
