#!/bin/bash
# =============================================================================
# Unified Benchmark Experiment Grid
# =============================================================================
# Runs benchmark experiments with varying context policies and init qualities.
# Tests how different context axes affect agent optimization performance.
#
# Init qualities (from landscape characterization):
#   - high.json   → good start    (r <= 0.20, top performers)
#   - neutral.json → general start (0.45 <= r <= 0.55, middle band)
#   - low.json    → bad start     (r >= 0.80, bottom performers)
#
# Prerequisites:
#   Run landscape characterization first to generate init configs:
#     python scripts/run_landscape.py --benchmark <benchmark> --n-configs 256
# =============================================================================

set -euo pipefail

# Require jq for JSON extraction
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt install jq (Linux)"
    exit 1
fi

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
    echo "  python scripts/run_landscape.py --benchmark <benchmark> --n-configs 256"
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
    echo "  python scripts/run_landscape.py --benchmark $BENCHMARK --n-configs 256"
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
TOTAL=144  # 3 × 2 × 2 × 2 × 2 × 3 = 144 runs (init_quality × fd × t × m × b × seed)
FAILED_CONFIGS=()

# Override file path (benchmark reads init_override.json if present, else config.json)
OVERRIDE_PATH="src/benchmarks/${BENCHMARK}/workspace/init_override.json"

# Cleanup: remove override file on exit so normal runs use config.json
cleanup() {
    rm -f "$OVERRIDE_PATH"
}
trap 'cleanup; echo -e "\nInterrupted. Exiting..."; exit 130' INT
trap cleanup EXIT

count=0
for init_quality in low neutral high; do
    # Extract config field from init JSON and write to override file
    # (init JSONs contain {config, score, normalized_regret, ...})
    jq '.config' "$INIT_CONFIG_DIR/$init_quality.json" > "$OVERRIDE_PATH"
    echo "=== Init quality: $init_quality (override written) ==="

    for feedback_depth in 1 5; do
        for show_task in 0 1; do
            for show_metric in 0 1; do
                for show_bounds in 0 1; do
                    for seed in 0 1 2; do
                        count=$((count + 1))

                        # Build run_id (includes timestamp subdir for grid organization)
                        run_id="${TIMESTAMP}/${BENCHMARK}_i${init_quality}_fd${feedback_depth}_t${show_task}_m${show_metric}_b${show_bounds}_s${seed}"

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
                            --seed $seed \
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
done

if ! $DRY_RUN && [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo ""
    echo "=== FAILED CONFIGS (${#FAILED_CONFIGS[@]}) ==="
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - $cfg"
    done
    exit 1
fi
