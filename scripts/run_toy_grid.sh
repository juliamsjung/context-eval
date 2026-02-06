#!/bin/bash
# =============================================================================
# Toy Benchmark Experiment Grid
# =============================================================================
# Runs logistic regression hyperparameter tuning with varying context policies.
# Tests how different context axes affect agent optimization performance.
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
NUM_STEPS=5
OUTPUT_DIR="traces"
TOTAL=48
DRY_RUN=false
FAILED_CONFIGS=()

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$OUTPUT_DIR"

count=0
for history_window in 0 5; do
    for show_task in 0 1; do
        for show_metric in 0 1; do
            for show_resources in 0 1; do
                for seed in 0 1 2; do
                    count=$((count + 1))

                    # Build run_id
                    run_id="toy_hw${history_window}_t${show_task}_m${show_metric}_r${show_resources}_s${seed}"

                    # Build flag strings
                    task_flag=""
                    metric_flag=""
                    resources_flag=""
                    [[ $show_task -eq 1 ]] && task_flag="--show-task"
                    [[ $show_metric -eq 1 ]] && metric_flag="--show-metric"
                    [[ $show_resources -eq 1 ]] && resources_flag="--show-resources"

                    cmd="$PYTHON run_toy_bench.py \
                        --num-steps $NUM_STEPS \
                        --history-window $history_window \
                        --seed $seed \
                        --run-id $run_id \
                        --output-dir $OUTPUT_DIR \
                        $task_flag $metric_flag $resources_flag"

                    if $DRY_RUN; then
                        echo "$cmd"
                    else
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$count / $TOTAL] Running: $run_id"
                        if ! (set +e; eval "$cmd"); then
                            echo "FAILED: $run_id"
                            FAILED_CONFIGS+=("$run_id")
                        fi
                    fi
                done
            done
        done
    done
done

if ! $DRY_RUN && [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo ""
    echo "=== FAILED CONFIGS (${#FAILED_CONFIGS[@]}) ==="
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - $cfg"
    done
    exit 1
fi
