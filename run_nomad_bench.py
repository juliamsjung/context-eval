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

    results = run_nomad_bench(
        num_steps=args.num_steps,
        history_window=args.history_window,
        show_task=args.show_task,
        show_metric=args.show_metric,
        show_resources=args.show_resources,
        seed=args.seed,
        run_id=args.run_id,
        model=args.model,
        temperature=args.temperature,
        debug_show_prompt=args.debug_show_prompt,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
