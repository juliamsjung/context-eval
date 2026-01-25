#!/usr/bin/env python3
"""CLI entry point for Toy benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.toy import run_toy_tabular


def main() -> None:
    args, cfg = parse_benchmark_args(
        description="Run Toy tabular benchmark",
        benchmark_config_key="toy_bench",
    )

    results = run_toy_tabular(
        num_steps=args.num_steps,
        policy_type=args.policy_type,
        reasoning_mode=args.reasoning_mode,
        config=cfg,
        seed=args.seed,
        run_id=args.run_id,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
