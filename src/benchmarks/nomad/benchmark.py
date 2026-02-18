"""NOMAD benchmark implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.benchmarks.base import (
    BaseBenchmark, BenchmarkConfig, _clamp,
    sanitize_with_clamp_tracking,
)
from src.benchmarks.nomad.env import NomadEnv


PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.5),
    "max_depth": (2, 16),
    "max_iter": (50, 1000),
    "l2_regularization": (0.0, 2.0),
    "max_leaf_nodes": (15, 255),
    "min_samples_leaf": (5, 200),
}


class NomadBenchmark(BaseBenchmark):
    """NOMAD materials science regression benchmark."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.env = NomadEnv()

    @property
    def benchmark_name(self) -> str:
        return "nomad"

    @property
    def dataset_id(self) -> str:
        return "nomad"

    @property
    def agent_id(self) -> str:
        return "nomad_llm"

    @property
    def workspace_path(self) -> Path:
        return Path(__file__).resolve().parent / "workspace"

    def get_default_config(self) -> Dict[str, Any]:
        return self.env.read_config()

    def run_training(self, config: Dict[str, Any]) -> Dict[str, float]:
        self.env.write_config(config)
        results = self.env.run_train()
        metrics = results.get("metrics", {})
        return {
            "rmsle": metrics.get("rmsle", 0.0),
            "mae": metrics.get("mae", 0.0),
            "rmse": metrics.get("rmse", 0.0),
            "r2": metrics.get("r2", 0.0),
        }

    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        factor = 1.15 if step % 2 == 0 else 0.85
        return {
            "learning_rate": _clamp(current_config.get("learning_rate", 0.1) * factor, PARAM_BOUNDS["learning_rate"]),
            "max_depth": int(
                _clamp(
                    current_config.get("max_depth", 5) + (1 if step % 2 == 0 else -1),
                    PARAM_BOUNDS["max_depth"],
                )
            ),
            "max_iter": int(
                _clamp(
                    current_config.get("max_iter", 100) + (50 if step % 2 == 0 else -30),
                    PARAM_BOUNDS["max_iter"],
                )
            ),
            "l2_regularization": _clamp(
                current_config.get("l2_regularization", 0.1) * (1.3 if step % 2 == 0 else 0.7),
                PARAM_BOUNDS["l2_regularization"],
            ),
            "max_leaf_nodes": int(
                _clamp(
                    current_config.get("max_leaf_nodes", 31) + (8 if step % 2 == 0 else -8),
                    PARAM_BOUNDS["max_leaf_nodes"],
                )
            ),
            "min_samples_leaf": int(
                _clamp(
                    current_config.get("min_samples_leaf", 20) + (-2 if step % 2 == 0 else 2),
                    PARAM_BOUNDS["min_samples_leaf"],
                )
            ),
        }

    def sanitize_config(self, proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        return sanitize_with_clamp_tracking(
            proposal, PARAM_BOUNDS,
            integer_keys={"max_depth", "max_leaf_nodes", "max_iter", "min_samples_leaf"},
        )

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback (lower is better for RMSLE)."""
        return metrics.get("rmsle", 0.0)

    def _is_higher_better(self) -> bool:
        """RMSLE: lower is better."""
        return False

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for this benchmark."""
        return PARAM_BOUNDS

    def _filter_config_for_prompt(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter config, excluding keys not present in config (preserves exact formatting)."""
        return {k: config.get(k) for k in self.param_bounds.keys() if k in config}

    def _get_task_intro(self) -> str:
        """Return task-specific introduction text."""
        return "You are tuning a HistGradientBoostingRegressor."

    def _get_output_format_instructions(self) -> str:
        """Return output format instructions."""
        return "Values must be numeric and within reasonable ranges."


def run_nomad_bench(args, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run NOMAD benchmark."""
    config = BenchmarkConfig.from_args(args)
    benchmark = NomadBenchmark(config)
    return benchmark.run(run_id=run_id or args.run_id)
