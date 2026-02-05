"""Shared CLI argument parsing utilities for benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils import logging as trace_logging


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
    parser.add_argument("--output-dir", type=str, help="Custom output directory for traces.")
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
    parser.add_argument("--history-window", type=int, default=5, help="Number of history entries to show (default: 5, 0=none).")
    
    args = parser.parse_args()
    
    # Override trace output directory if specified
    if args.output_dir:
        trace_logging.TRACES_ROOT = Path(args.output_dir)

    return args

