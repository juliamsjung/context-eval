#!/usr/bin/env python3
"""CLI entry point for NOMAD benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import load_config
from src.benchmarks.nomad import run_nomad_bench
from src.utils import logging as trace_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NOMAD benchmark")
    parser.add_argument("--config", default="config.json", help="Path to the main project config.")
    parser.add_argument("--num-steps", type=int, help="Override number of iterations.")
    parser.add_argument(
        "--history-window",
        type=int,
        help="Number of previous entries to expose to the LLM prompt.",
    )
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
    # Batch experiment flags
    parser.add_argument("--output-dir", type=str, help="Custom output directory for traces.")
    parser.add_argument("--run-id", type=str, help="Custom run ID for batch tracking.")
    args = parser.parse_args()

    # Override trace output directory if specified
    if args.output_dir:
        trace_logging.TRACES_ROOT = Path(args.output_dir)

    cfg = load_config(args.config)
    bench_cfg = cfg.get("nomad_bench", {})
    num_steps = args.num_steps or int(bench_cfg.get("num_steps", 3))
    history_window = args.history_window or int(bench_cfg.get("history_window", 5))
    policy_type = args.policy_type or cfg.get("policy_type", "short_context")
    reasoning_mode = args.reasoning_mode or cfg.get("reasoning_mode", "controller")

    results = run_nomad_bench(
        num_steps=num_steps,
        history_window=history_window,
        policy_type=policy_type,
        reasoning_mode=reasoning_mode,
        config=cfg,
        seed=args.seed,
        run_id=args.run_id,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
