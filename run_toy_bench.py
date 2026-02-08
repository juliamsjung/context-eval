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

    results = run_toy_tabular(args)

    if args.verbose:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
