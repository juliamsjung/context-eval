"""Forest Cover Type Prediction benchmark module."""
from src.benchmarks.forest.benchmark import ForestBenchmark, run_forest_bench
from src.benchmarks.forest.env import ForestEnv

__all__ = ["ForestBenchmark", "ForestEnv", "run_forest_bench"]
