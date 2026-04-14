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

run_bench_for_config "$@" impl=seq
