"""Shared CLI argument parsing utilities for benchmarks."""
from __future__ import annotations

import argparse



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

    # Experimental controls (model, temperature)
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="LLM temperature (default: 0)")

    # Context visibility flags
    parser.add_argument("--show-task", action="store_true", help="Include task description in prompt.")
    parser.add_argument("--show-metric", action="store_true", help="Include metric description in prompt.")
    parser.add_argument("--show-resources", action="store_true", help="Include resource usage (tokens, cost, latency) in prompt.")
    parser.add_argument("--history-window", type=int, default=0, help="Number of history entries to show (default: 0, i.e., none).")

    # Developer tools
    parser.add_argument("--debug-show-llm", action="store_true", help="Print full LLM request and response for debugging.")
    parser.add_argument("--verbose", action="store_true", help="Enable step-by-step logging.")

    return parser.parse_args()

