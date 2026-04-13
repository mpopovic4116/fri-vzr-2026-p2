#!/bin/bash

OUT_FILE="results.txt"
TRIALS=1
echo "Lenia bench results - $(date)" > $OUT_FILE
echo "------------------------------------------------" >> $OUT_FILE

RUNS=(
    "ker=default,tor=y,unroll=y, thr=8x8|cuda_arch=v100s precision=32 impl=cuda toroid=bitwise unroll=y kernel=default threads_x=8 threads_y=8 size_w=4096 gif=1.gif"
    "ker=default,tor=y,unroll=y, thr=8x16|cuda_arch=v100s precision=32 impl=cuda toroid=bitwise unroll=y kernel=default threads_x=8 threads_y=16 size_w=4096 gif=2.gif"
    "ker=default,tor=y,unroll=y,thr=16x16|cuda_arch=v100s precision=32 impl=cuda toroid=bitwise unroll=y kernel=default threads_x=16 threads_y=16 size_w=4096 gif=3.gif"
    "ker=default,tor=y,unroll=y,thr=16x32|cuda_arch=v100s precision=32 impl=cuda toroid=bitwise unroll=y kernel=default threads_x=16 threads_y=32 size_w=4096 gif=4.gif"
    "ker=default,tor=y,unroll=y,thr=32x32|cuda_arch=v100s precision=32 impl=cuda toroid=bitwise unroll=y kernel=default threads_x=32 threads_y=32 size_w=4096 gif=5.gif"
)


for run in "${RUNS[@]}"; do
    IFS="|" read -r label args <<< "$run"
    echo "Running $label..."
    echo "## Scenario: $label" >> $OUT_FILE
    
    t_work_total=0
    for i in $(seq 1 $TRIALS); do
        output=$(./scripts/run.sh $args 2>/dev/null | grep "t_work_total")
        if [ -n "$output" ]; then
            echo "  Trial $i: $output" >> $OUT_FILE
        fi
    done
    echo "" >> $OUT_FILE
done

echo "Done. Out file: $OUT_FILE"