#!/usr/bin/env python3
"""CLI entry point for NOMAD benchmark."""
from __future__ import annotations

import json

from src.utils.cli import parse_benchmark_args
from src.benchmarks.nomad import run_nomad_bench


def main() -> None:
    args, cfg = parse_benchmark_args(
        description="Run NOMAD benchmark",
        benchmark_config_key="nomad_bench",
    )

    # Get history_window from args or config default
    bench_cfg = cfg.get("nomad_bench", {})
    history_window = args.history_window if args.history_window is not None else int(bench_cfg.get("history_window", 5))

    results = run_nomad_bench(
        num_steps=args.num_steps,
        history_window=history_window,
        show_task=args.show_task,
        show_metric=args.show_metric,
        config=cfg,
        seed=args.seed,
        run_id=args.run_id,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
