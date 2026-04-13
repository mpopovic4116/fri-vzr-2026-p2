#!/usr/bin/env bash

# Thin wrapper around srun with the correct args.
# Won't work outside the cluster, obviously.
# Usage: scripts/srun.sh <--cpu|--gpu> [--cpus=<n>] args...

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/..

cpu=
gpu=
cpus=1

while (( $# )); do
    case "$1" in
        --gpu)
            gpu=y
            cpu=
            shift
            ;;
        --cpu)
            gpu=
            cpu=y
            shift
            ;;
        --cpus=*)
            cpus=${1#*=}
            shift
            ;;
        *) break ;;
    esac
done

if [ -z "$cpu" ] && [ -z "$gpu" ]; then
    echo "Must use --gpu or --cpu" >&2
    exit 1
fi

if ! [[ "$cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "--cpus=<n> must be a positive int"
    exit 1
fi

flags=(
    --reservation=fri
    --job-name=vzr-g17-p2
    --ntasks=1
    --cpus-per-task="$cpus"
    --hint=nomultithread
)

if [ -n "$gpu" ]; then
    flags+=(
        --partition=gpu
        --gpus=1
    )
fi

srun "${flags[@]}" -- "$@"
