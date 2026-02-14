#!/bin/bash
# =============================================================================
# Unified Benchmark Experiment Grid
# =============================================================================
# Runs benchmark experiments with varying context policies.
# Tests how different context axes affect agent optimization performance.
# =============================================================================

set -euo pipefail

# Exit on Ctrl+C
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

# Show usage
usage() {
    echo "Usage: $0 <benchmark> [--num-steps N] [--dry-run]"
    echo ""
    echo "Arguments:"
    echo "  benchmark     One of: toy, nomad, jigsaw, leaf"
    echo ""
    echo "Options:"
    echo "  --num-steps N  Number of steps per run (default: 10)"
    echo "  --dry-run      Print commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 toy                    # Run toy benchmark with 10 steps"
    echo "  $0 nomad --num-steps 20   # Run nomad with 20 steps"
    echo "  $0 toy --dry-run          # Print commands without running"
    exit 1
}

# Require benchmark argument
if [[ $# -lt 1 ]]; then
    usage
fi

BENCHMARK="$1"
NUM_STEPS=10  # Default for all benchmarks
DRY_RUN=false

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate benchmark
case "$BENCHMARK" in
    toy|nomad|jigsaw|leaf) ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Available: toy, nomad, jigsaw, leaf"
        exit 1
        ;;
esac

# Derived variables
PYTHON="${PYTHON:-python}"
PYTHON_SCRIPT="run_${BENCHMARK}_bench.py"
TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
EXPERIMENT_ID="grid_${TIMESTAMP}"
TOTAL=96
FAILED_CONFIGS=()

count=0
for feedback_depth in 1 5; do
    for show_task in 0 1; do
        for show_metric in 0 1; do
            for show_resources in 0 1; do
                for show_diagnostics in 0 1; do
                    for seed in 0 1 2; do
                        count=$((count + 1))

                        # Build run_id (includes timestamp subdir for grid organization)
                        run_id="${TIMESTAMP}/${BENCHMARK}_fd${feedback_depth}_t${show_task}_m${show_metric}_r${show_resources}_d${show_diagnostics}_s${seed}"

                        # Build flag strings
                        task_flag=""
                        metric_flag=""
                        resources_flag=""
                        diagnostics_flag=""
                        [[ $show_task -eq 1 ]] && task_flag="--show-task"
                        [[ $show_metric -eq 1 ]] && metric_flag="--show-metric"
                        [[ $show_resources -eq 1 ]] && resources_flag="--show-resources"
                        [[ $show_diagnostics -eq 1 ]] && diagnostics_flag="--show-diagnostics"

                        cmd="$PYTHON $PYTHON_SCRIPT \
                            --num-steps $NUM_STEPS \
                            --feedback-depth $feedback_depth \
                            --seed $seed \
                            --run-id $run_id \
                            --experiment-id $EXPERIMENT_ID \
                            $task_flag $metric_flag $resources_flag $diagnostics_flag"

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
done

if ! $DRY_RUN && [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo ""
    echo "=== FAILED CONFIGS (${#FAILED_CONFIGS[@]}) ==="
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - $cfg"
    done
    exit 1
fi
