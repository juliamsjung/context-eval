#!/bin/bash
# =============================================================================
# Unified Benchmark Experiment Grid
# =============================================================================
# Runs benchmark experiments with varying context policies and init qualities.
# Tests how different context axes affect agent optimization performance.
#
# Prerequisites:
#   Run landscape characterization first to generate init configs:
#     python scripts/run_landscape.py --benchmark <benchmark> --num-samples 200
# =============================================================================

set -euo pipefail

# Exit on Ctrl+C
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

# Show usage
usage() {
    echo "Usage: $0 <benchmark> [--num-steps N] [--dry-run]"
    echo ""
    echo "Arguments:"
    echo "  benchmark     One of: toy, nomad, jigsaw, forest, housing"
    echo ""
    echo "Options:"
    echo "  --num-steps N  Number of steps per run (default: 10)"
    echo "  --dry-run      Print commands without executing"
    echo ""
    echo "Prerequisites:"
    echo "  python scripts/run_landscape.py --benchmark <benchmark>"
    echo ""
    echo "Examples:"
    echo "  $0 nomad                   # Run nomad with 10 steps"
    echo "  $0 nomad --num-steps 20    # Run nomad with 20 steps"
    echo "  $0 nomad --dry-run         # Print commands without running"
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
    toy|nomad|jigsaw|forest|housing) ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Available: toy, nomad, jigsaw, forest, housing"
        exit 1
        ;;
esac

# Validate that landscape init configs exist
INIT_CONFIG_DIR="logs/landscape/${BENCHMARK}_init_configs"
if [[ ! -d "$INIT_CONFIG_DIR" ]]; then
    echo "ERROR: Init configs not found at $INIT_CONFIG_DIR"
    echo "Run landscape characterization first:"
    echo "  python scripts/run_landscape.py --benchmark $BENCHMARK --num-samples 200"
    exit 1
fi

for quality in low neutral high; do
    if [[ ! -f "$INIT_CONFIG_DIR/$quality.json" ]]; then
        echo "ERROR: Missing init config: $INIT_CONFIG_DIR/$quality.json"
        exit 1
    fi
done

# Derived variables
PYTHON="${PYTHON:-python}"
PYTHON_SCRIPT="run_${BENCHMARK}_bench.py"
TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
EXPERIMENT_ID="grid_${TIMESTAMP}"
TOTAL=48  # 3 × 2 × 2 × 2 × 2 = 48 runs
FAILED_CONFIGS=()

# Save original config.json for restoration
CONFIG_PATH="src/benchmarks/${BENCHMARK}/workspace/config.json"
OVERRIDE_PATH="src/benchmarks/${BENCHMARK}/workspace/init_override.json"

# Cleanup: remove override file on exit so normal runs use config.json
cleanup() {
    rm -f "$OVERRIDE_PATH"
}
trap cleanup EXIT
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

count=0
for init_quality in low neutral high; do
    # Write override file (config.json is never touched)
    cp "$INIT_CONFIG_DIR/$init_quality.json" "$OVERRIDE_PATH"
    echo "=== Init quality: $init_quality (override written) ==="

    for feedback_depth in 1 5; do
        for show_task in 0 1; do
            for show_metric in 0 1; do
                for show_bounds in 0 1; do
                    count=$((count + 1))

                    # Build run_id (includes timestamp subdir for grid organization)
                    run_id="${TIMESTAMP}/${BENCHMARK}_i${init_quality}_fd${feedback_depth}_t${show_task}_m${show_metric}_b${show_bounds}"

                    # Build flag strings
                    task_flag=""
                    metric_flag=""
                    bounds_flag=""
                    [[ $show_task -eq 1 ]] && task_flag="--show-task"
                    [[ $show_metric -eq 1 ]] && metric_flag="--show-metric"
                    [[ $show_bounds -eq 1 ]] && bounds_flag="--show-bounds"

                    cmd="$PYTHON $PYTHON_SCRIPT \
                        --num-steps $NUM_STEPS \
                        --feedback-depth $feedback_depth \
                        --seed 0 \
                        --run-id $run_id \
                        --experiment-id $EXPERIMENT_ID \
                        $task_flag $metric_flag $bounds_flag"

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
