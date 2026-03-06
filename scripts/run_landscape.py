#!/usr/bin/env python3
"""Run landscape characterization for a benchmark.

Generates space-filling samples via Sobol sequences, evaluates each
against the benchmark's training pipeline, and selects stratified
init configs (good/general/bad strata) for controlled experiments.

Usage:
    python scripts/run_landscape.py --benchmark nomad [--n-configs 256] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landscape.sampler import SobolSampler
from src.landscape.runner import LandscapeRunner
from src.landscape.selector import StratifiedSelector


# Registry of benchmark metadata needed for landscape characterization.
# Each entry uses the benchmark module's constants directly.
BENCHMARK_REGISTRY = {}


def _register_benchmarks() -> None:
    """Lazily register benchmarks to avoid import errors if data is missing."""
    global BENCHMARK_REGISTRY

    from src.benchmarks.nomad.benchmark import (
        NomadBenchmark, PARAM_BOUNDS as NOMAD_BOUNDS,
        LOG_SCALE_PARAMS as NOMAD_LOG, INTEGER_KEYS as NOMAD_INT,
    )
    from src.benchmarks.jigsaw.benchmark import (
        JigsawBenchmark, PARAM_BOUNDS as JIGSAW_BOUNDS,
        LOG_SCALE_PARAMS as JIGSAW_LOG, INTEGER_KEYS as JIGSAW_INT,
    )
    from src.benchmarks.forest.benchmark import (
        ForestBenchmark, PARAM_BOUNDS as FOREST_BOUNDS,
        LOG_SCALE_PARAMS as FOREST_LOG, INTEGER_KEYS as FOREST_INT,
    )
    from src.benchmarks.housing.benchmark import (
        HousingBenchmark, PARAM_BOUNDS as HOUSING_BOUNDS,
        LOG_SCALE_PARAMS as HOUSING_LOG, INTEGER_KEYS as HOUSING_INT,
    )
    from src.benchmarks.base import BenchmarkConfig

    # Minimal config — no LLM needed for landscape evaluation
    dummy_config = BenchmarkConfig(num_steps=0, verbose=False)

    BENCHMARK_REGISTRY.update({
        "nomad": {
            "benchmark_class": NomadBenchmark,
            "config": dummy_config,
            "param_bounds": NOMAD_BOUNDS,
            "log_scale_params": NOMAD_LOG,
            "integer_keys": NOMAD_INT,
            "higher_is_better": False,  # RMSLE: lower is better
        },
        "jigsaw": {
            "benchmark_class": JigsawBenchmark,
            "config": dummy_config,
            "param_bounds": JIGSAW_BOUNDS,
            "log_scale_params": JIGSAW_LOG,
            "integer_keys": JIGSAW_INT,
            "higher_is_better": True,  # AUC: higher is better
        },
        "forest": {
            "benchmark_class": ForestBenchmark,
            "config": dummy_config,
            "param_bounds": FOREST_BOUNDS,
            "log_scale_params": FOREST_LOG,
            "integer_keys": FOREST_INT,
            "higher_is_better": True,  # Accuracy: higher is better
        },
        "housing": {
            "benchmark_class": HousingBenchmark,
            "config": dummy_config,
            "param_bounds": HOUSING_BOUNDS,
            "log_scale_params": HOUSING_LOG,
            "integer_keys": HOUSING_INT,
            "higher_is_better": False,  # RMSE: lower is better
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run landscape characterization for a benchmark.",
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=["nomad", "jigsaw", "forest", "housing"],
        help="Benchmark to characterize.",
    )
    parser.add_argument(
        "--n-configs", type=int, default=256,
        help="Number of Sobol samples to evaluate (default: 256, power of 2 for exact balance).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for Sobol sequence (default: 0).",
    )
    args = parser.parse_args()

    _register_benchmarks()
    entry = BENCHMARK_REGISTRY[args.benchmark]

    print(f"=== Landscape Characterization: {args.benchmark} ===")
    print(f"Configs: {args.n_configs}, Seed: {args.seed}")
    print()

    # Step 1: Generate Sobol samples
    sampler = SobolSampler(
        param_bounds=entry["param_bounds"],
        log_scale_params=entry["log_scale_params"],
        integer_keys=entry["integer_keys"],
        seed=args.seed,
    )
    configs = sampler.sample(n=args.n_configs)
    print(f"Generated {len(configs)} Sobol samples.")

    # Step 2: Instantiate benchmark and evaluate
    benchmark = entry["benchmark_class"](entry["config"])

    runner = LandscapeRunner(
        run_training=benchmark.run_training,
        score_extractor=benchmark._get_primary_score,
        higher_is_better=entry["higher_is_better"],
        benchmark_name=args.benchmark,
    )
    results = runner.evaluate(configs)

    successful = sum(1 for r in results if r["error"] is None)
    print(f"\nEvaluated {successful}/{len(configs)} configs successfully.")

    # Step 3: Select stratified init configs
    selector = StratifiedSelector(higher_is_better=entry["higher_is_better"])
    selections = selector.select(
        results,
        benchmark_name=args.benchmark,
    )

    print("\n=== Stratified Init Configs ===")
    for quality in ["high", "neutral", "low"]:
        data = selections[quality]
        print(f"  {quality:>8s}: score={data['score']:.4f}, r={data['normalized_regret']:.4f}")

    output_dir = Path("logs/landscape") / f"{args.benchmark}_init_configs"
    print(f"\nInit configs saved to: {output_dir}/")
    print("Landscape results saved to: logs/landscape/")


if __name__ == "__main__":
    main()
