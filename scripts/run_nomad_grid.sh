#!/bin/bash
# =============================================================================
# NOMAD Benchmark Experiment Grid
# =============================================================================
# Runs materials science regression (bandgap/formation energy prediction)
# with varying context policies. Tests agent performance on real Kaggle data.
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
NUM_STEPS=20
EXPERIMENT_NAME="nomad_context_axes"
TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
OUTPUT_DIR="traces/${EXPERIMENT_NAME}/${TIMESTAMP}"
TOTAL=48
DRY_RUN=false
FAILED_CONFIGS=()

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$OUTPUT_DIR"

# Generate experiment README
cat > "$OUTPUT_DIR/README.md" << EOF
# NOMAD - Context Axes Experiment

Timestamp (UTC): $TIMESTAMP

Benchmark: Materials science regression (bandgap/formation energy)

Axes varied:
- history_window: [0, 5]
- show_task: [false, true]
- show_metric: [false, true]
- show_resources: [false, true]

Fixed:
- model = gpt-4o-mini
- temperature = 0
- steps = $NUM_STEPS
- seeds = [0, 1, 2]

Total runs: $TOTAL
EOF

count=0
for history_window in 0 5; do
    for show_task in 0 1; do
        for show_metric in 0 1; do
            for show_resources in 0 1; do
                for seed in 0 1 2; do
                    count=$((count + 1))

                    # Build run_id
                    run_id="nomad_hw${history_window}_t${show_task}_m${show_metric}_r${show_resources}_s${seed}"

                    # Build flag strings
                    task_flag=""
                    metric_flag=""
                    resources_flag=""
                    [[ $show_task -eq 1 ]] && task_flag="--show-task"
                    [[ $show_metric -eq 1 ]] && metric_flag="--show-metric"
                    [[ $show_resources -eq 1 ]] && resources_flag="--show-resources"

                    cmd="$PYTHON run_nomad_bench.py \
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
