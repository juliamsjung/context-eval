#!/bin/bash
# =============================================================================
# Random Search Baseline
# =============================================================================
# Runs random search baseline for comparison with LLM optimizer.
# Random search ignores context axes, so we only vary init quality and seed.
#
# Runs per benchmark: 3 init × 3 seeds = 9
# Total: 27 runs (3 benchmarks)
# Cost: Zero API cost, ~5 min total runtime
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
    echo "Usage: $0 [--benchmarks <list>] [--num-steps N] [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --benchmarks   Comma-separated list (default: nomad,forest,housing)"
    echo "  --num-steps N  Number of steps per run (default: 10)"
    echo "  --dry-run      Print commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run all benchmarks"
    echo "  $0 --benchmarks nomad           # Run only nomad"
    echo "  $0 --benchmarks nomad,forest    # Run nomad and forest"
    echo "  $0 --dry-run                    # Preview commands"
    exit 1
}

# Defaults
BENCHMARKS="nomad,forest,housing"
NUM_STEPS=10
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Convert comma-separated to array
IFS=',' read -ra BENCHMARK_ARRAY <<< "$BENCHMARKS"

PYTHON="${PYTHON:-python}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
EXPERIMENT_ID="random_baseline_${TIMESTAMP}"
FAILED_RUNS=()

echo "=== Random Search Baseline ==="
echo "Benchmarks: ${BENCHMARK_ARRAY[*]}"
echo "Steps: $NUM_STEPS"
echo "Experiment ID: $EXPERIMENT_ID"
echo ""

count=0
total=$((${#BENCHMARK_ARRAY[@]} * 3 * 3))  # benchmarks × init × seeds

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
    INIT_CONFIG_DIR="logs/landscape/${benchmark}_init_configs"

    # Validate init configs exist
    if [[ ! -d "$INIT_CONFIG_DIR" ]]; then
        echo "ERROR: Init configs not found at $INIT_CONFIG_DIR"
        echo "Run landscape characterization first:"
        echo "  python scripts/run_landscape.py --benchmark $benchmark --n-configs 256"
        exit 1
    fi

    PYTHON_SCRIPT="run_${benchmark}_bench.py"
    OVERRIDE_PATH="src/benchmarks/${benchmark}/workspace/init_override.json"

    # Cleanup override file on exit
    cleanup() {
        rm -f "$OVERRIDE_PATH"
    }
    trap 'cleanup; echo -e "\nInterrupted. Exiting..."; exit 130' INT
    trap cleanup EXIT

    for init_quality in low neutral high; do
        # Extract config from init JSON and write to override file
        init_source="$INIT_CONFIG_DIR/$init_quality.json"
        echo "=== Init: $init_quality (from $init_source) ==="
        jq '.config' "$init_source" > "$OVERRIDE_PATH"

        for seed in 0 1 2; do
            count=$((count + 1))
            run_id="${TIMESTAMP}/${benchmark}_random_i${init_quality}_s${seed}"

            cmd="$PYTHON $PYTHON_SCRIPT \
                --num-steps $NUM_STEPS \
                --optimizer random \
                --seed $seed \
                --run-id $run_id \
                --experiment-id $EXPERIMENT_ID"

            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$count / $total] Running: $run_id"
                if ! (set +e; eval "$cmd"); then
                    echo "FAILED: $run_id"
                    FAILED_RUNS+=("$run_id")
                fi
            fi
        done
    done

    # Clean up override after each benchmark
    rm -f "$OVERRIDE_PATH"
done

if ! $DRY_RUN; then
    echo ""
    echo "=== Complete ==="
    echo "Results saved to: logs/runs/$EXPERIMENT_ID/"

    if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
        echo ""
        echo "=== FAILED RUNS (${#FAILED_RUNS[@]}) ==="
        for run in "${FAILED_RUNS[@]}"; do
            echo "  - $run"
        done
        exit 1
    fi
fi
