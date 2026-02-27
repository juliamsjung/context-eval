#!/usr/bin/env python3
"""CLI entry point for Forest Cover Type Prediction benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.forest import run_forest_bench


def main() -> None:
    args = parse_benchmark_args(
        description="Run Forest Cover Type Prediction benchmark",
    )

    results = run_forest_bench(args)

    if args.verbose:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
