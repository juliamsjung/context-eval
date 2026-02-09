#!/usr/bin/env python3
"""CLI entry point for NOMAD benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.nomad import run_nomad_bench


def main() -> None:
    args = parse_benchmark_args(
        description="Run NOMAD benchmark",
    )

    results = run_nomad_bench(args)

    if args.verbose:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
