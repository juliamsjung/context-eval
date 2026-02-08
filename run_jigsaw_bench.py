#!/usr/bin/env python3
"""CLI entry point for Jigsaw Toxic Comment Classification benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.jigsaw import run_jigsaw_bench


def main() -> None:
    args = parse_benchmark_args(
        description="Run Jigsaw Toxic Comment Classification benchmark",
    )

    results = run_jigsaw_bench(args)

    if args.verbose:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
