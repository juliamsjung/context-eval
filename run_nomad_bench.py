#!/usr/bin/env python3
"""
Runner for the NOMAD Kaggle benchmark with Phoenix tracing.
"""
from __future__ import annotations

import argparse
import json

from code import PhoenixTracerManager, _phoenix_settings, load_config
from benchmarks.nomad.agent import run_nomad_bench


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NOMAD bench with Phoenix tracing.")
    parser.add_argument("--config", default="config.json", help="Path to the main project config.")
    parser.add_argument("--num-steps", type=int, help="Override number of iterations.")
    parser.add_argument(
        "--history-window",
        type=int,
        help="Number of previous entries to expose to the LLM prompt.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench_cfg = cfg.get("nomad_bench", {})
    num_steps = args.num_steps or int(bench_cfg.get("num_steps", 3))
    history_window = args.history_window or int(bench_cfg.get("history_window", 5))

    tracer = PhoenixTracerManager(_phoenix_settings(cfg))
    span_cm = tracer.span(
        "nomad.bench.run",
        {
            "benchmark": "nomad",
            "dataset": "nomad2018-predict-transparent-conductors",
            "num_steps": num_steps,
            "history_window": history_window,
        },
    )
    try:
        with span_cm as span:
            results = run_nomad_bench(
                num_steps=num_steps,
                tracer=tracer,
                history_window=history_window,
            )
            if span:
                tracer.set_attributes(
                    span,
                    {
                        "final_metric": results.get("final_metric"),
                        "final_metric_name": results.get("final_metric_name"),
                        "history_length": len(results.get("history", [])),
                    }
                    | {
                        f"final_config.{k}": v for k, v in results.get("final_config", {}).items()
                    },
                )
    finally:
        tracer.shutdown()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

