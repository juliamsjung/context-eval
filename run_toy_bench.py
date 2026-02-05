#!/usr/bin/env python3
"""CLI entry point for Toy benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.toy import run_toy_tabular


def main() -> None:
    args = parse_benchmark_args(
        description="Run Toy tabular benchmark",
    )

    results = run_toy_tabular(
        num_steps=args.num_steps,
        history_window=args.history_window,
        show_task=args.show_task,
        show_metric=args.show_metric,
        show_resources=args.show_resources,
        seed=args.seed,
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
