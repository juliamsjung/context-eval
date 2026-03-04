#!/usr/bin/env python3
"""CLI entry point for California Housing Prediction benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.housing import run_housing_bench


def main() -> None:
    args = parse_benchmark_args(
        description="Run California Housing Prediction benchmark",
    )

    results = run_housing_bench(args)

    if args.verbose:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
