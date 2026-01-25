"""Shared CLI argument parsing utilities for benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import load_config
from src.utils import logging as trace_logging


def parse_benchmark_args(
    description: str,
    benchmark_config_key: str,
    extra_args: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse common benchmark CLI arguments with optional benchmark-specific arguments.

    Args:
        description: Description for the ArgumentParser
        benchmark_config_key: Key in config.json for benchmark-specific settings (e.g., "toy_bench", "nomad_bench")
        extra_args: Optional list of (arg_name, arg_kwargs) tuples for benchmark-specific arguments.
                    Example: [("--history-window", {"type": int, "help": "..."})]

    Returns:
        Tuple of (parsed_args, config_dict) where:
        - parsed_args: argparse.Namespace with all parsed arguments
        - config_dict: Loaded and processed config dictionary
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments
    parser.add_argument("--config", default="config.json", help="Path to the main project config.")
    parser.add_argument("--num-steps", type=int, help="Override number of tuning steps.")
    parser.add_argument(
        "--policy-type",
        choices=["short_context", "long_context"],
        help="Select the context policy for this run.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["controller", "agentic"],
        help="Toggle the reasoning style (agentic loop vs legacy controller).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, help="Custom output directory for traces.")
    parser.add_argument("--run-id", type=str, help="Custom run ID for batch tracking.")
    
    # Add benchmark-specific arguments if provided
    if extra_args:
        for arg_name, arg_kwargs in extra_args:
            parser.add_argument(arg_name, **arg_kwargs)
    
    args = parser.parse_args()
    
    # Override trace output directory if specified
    if args.output_dir:
        trace_logging.TRACES_ROOT = Path(args.output_dir)
    
    # Load and process config
    cfg = load_config(args.config)
    bench_cfg = cfg.get(benchmark_config_key, {})
    
    # Process num_steps with benchmark-specific default
    default_steps = int(bench_cfg.get("num_steps", 3))
    args.num_steps = args.num_steps or default_steps
    
    # Set defaults from config if not provided via CLI
    if args.policy_type is None:
        args.policy_type = cfg.get("policy_type", "short_context")
    if args.reasoning_mode is None:
        args.reasoning_mode = cfg.get("reasoning_mode", "controller")
    
    return args, cfg

