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
        for optimizations in y ''; do
            if [ "$kernel" = default ] && [ -n "$optimizations" ]; then
                continue
            fi
            if [ -n "$optimizations" ]; then
                toroid=bitwise
                unroll=y
            else
                toroid=mod
                unroll=
            fi
            for block in 16x16 16x64 64x16 32x32; do
                cuda_bench precision="$precision" kernel="$kernel" toroid="$toroid" unroll="$unroll" threads_x="${block%x*}" threads_y="${block#*x}"
            done
        done
    done
done
