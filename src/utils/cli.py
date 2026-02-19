"""Shared CLI argument parsing utilities for benchmarks."""
from __future__ import annotations

import argparse
import re
import sys



def parse_benchmark_args(
    description: str,
) -> argparse.Namespace:
    """
    Parse common benchmark CLI arguments.

    Args:
        description: Description for the ArgumentParser

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments
    parser.add_argument("--num-steps", type=int, default=3, help="Number of tuning steps (default: 3).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--run-id", type=str, help="Custom run ID for batch tracking.")
    parser.add_argument("--experiment-id", type=str, default="default",
                        help="Experiment ID for grouping runs (default: default). Must be filesystem-safe: [a-zA-Z0-9_-]+")

    # Experimental controls (model, temperature)
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="LLM temperature (default: 0)")

    # Context visibility flags
    parser.add_argument("--show-task", action="store_true", help="Include task description in prompt.")
    parser.add_argument("--show-metric", action="store_true", help="Include metric description in prompt.")
    parser.add_argument("--show-bounds", action="store_true", help="Include parameter bounds (valid ranges) in prompt.")
    parser.add_argument("--feedback-depth", type=int, default=1,
                        help="Feedback depth: number of visible outcome signals (1=current only, 5=current+4 history).")

    # Developer tools
    parser.add_argument("--debug-show-llm", action="store_true", help="Print full LLM request and response for debugging.")
    parser.add_argument("--debug-show-diff", action="store_true", help="Show config changes at each step.")
    parser.add_argument("--verbose", action="store_true", help="Enable step-by-step logging.")

    args = parser.parse_args()

    # Validate experiment_id is filesystem-safe
    if not re.match(r"^[a-zA-Z0-9_\-]+$", args.experiment_id):
        print(f"Error: --experiment-id must match [a-zA-Z0-9_-]+, got: {args.experiment_id!r}", file=sys.stderr)
        sys.exit(1)

    return args

