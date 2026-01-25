#!/bin/bash
# experiments/batch_run.sh - Run all experimental conditions
#
# This script runs all 160 experiments (2 benchmarks x 2 policies x 2 modes x 10 seeds)
#
# Usage:
#   ./experiments/batch_run.sh
#   ./experiments/batch_run.sh --num-steps 5 --seeds 5

set -e

# Default values
NUM_STEPS=${NUM_STEPS:-3}
NUM_SEEDS=${NUM_SEEDS:-10}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

OUTPUT_DIR="outputs/experiments/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/traces" "$OUTPUT_DIR/results"

echo "Starting batch experiments"
echo "Output directory: $OUTPUT_DIR"
echo "Number of steps: $NUM_STEPS"
echo "Number of seeds: $NUM_SEEDS"
echo ""

TOTAL_RUNS=$((2 * 2 * 2 * NUM_SEEDS))
CURRENT_RUN=0

for benchmark in toy nomad; do
    for policy in short_context long_context; do
        for mode in controller agentic; do
            for seed in $(seq 0 $((NUM_SEEDS - 1))); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                run_id="${benchmark}_${policy}_${mode}_seed${seed}"
                echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $run_id"

                if [ "$benchmark" = "nomad" ]; then
                    python run_nomad_bench.py \
                        --config config.json \
                        --num-steps $NUM_STEPS \
                        --policy-type $policy \
                        --reasoning-mode $mode \
                        --seed $seed \
                        --output-dir "$OUTPUT_DIR/traces" \
                        --run-id "$run_id" \
                        > "${OUTPUT_DIR}/results/${run_id}.json" 2>&1 || echo "FAILED: $run_id"
                else
                    python run_toy_bench.py \
                        --config config.json \
                        --num-steps $NUM_STEPS \
                        --policy-type $policy \
                        --reasoning-mode $mode \
                        --seed $seed \
                        --output-dir "$OUTPUT_DIR/traces" \
                        --run-id "$run_id" \
                        > "${OUTPUT_DIR}/results/${run_id}.json" 2>&1 || echo "FAILED: $run_id"
                fi
            done
        done
    done
done

echo ""
echo "All experiments complete!"
echo "Traces: $OUTPUT_DIR/traces/{nomad,toy_tabular}/*.jsonl"
echo "Results: $OUTPUT_DIR/results/*.json"
