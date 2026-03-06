"""California Housing regression benchmark implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.benchmarks.base import (
    BaseBenchmark, BenchmarkConfig, _clamp,
    sanitize_with_clamp_tracking,
)
from src.benchmarks.housing.env import HousingEnv


PARAM_BOUNDS = {
    "n_estimators": (10, 1500),
    "max_depth": (2, 80),
    "min_samples_split": (2, 80),
    "min_samples_leaf": (1, 60),
    "max_features": (0.05, 1.0),
    "bootstrap": (0, 1),
}

LOG_SCALE_PARAMS: set = set()  # No log-scale params for ExtraTrees
INTEGER_KEYS = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap"}


class HousingBenchmark(BaseBenchmark):
    """California Housing regression benchmark using ExtraTreesRegressor."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.env = HousingEnv()

    @property
    def benchmark_name(self) -> str:
        return "housing"

    @property
    def dataset_id(self) -> str:
        return "housing"

    @property
    def agent_id(self) -> str:
        return "housing_llm"

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
            "rmse": metrics.get("rmse", 0.0),
            "mae": metrics.get("mae", 0.0),
            "r2": metrics.get("r2", 0.0),
        }

    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        factor = 1.15 if step % 2 == 0 else 0.85
        return {
            "n_estimators": int(
                _clamp(
                    current_config.get("n_estimators", 100) * factor,
                    PARAM_BOUNDS["n_estimators"],
                )
            ),
            "max_depth": int(
                _clamp(
                    current_config.get("max_depth", 10) + (2 if step % 2 == 0 else -2),
                    PARAM_BOUNDS["max_depth"],
                )
            ),
            "min_samples_split": int(
                _clamp(
                    current_config.get("min_samples_split", 5) + (2 if step % 2 == 0 else -1),
                    PARAM_BOUNDS["min_samples_split"],
                )
            ),
            "min_samples_leaf": int(
                _clamp(
                    current_config.get("min_samples_leaf", 2) + (1 if step % 2 == 0 else -1),
                    PARAM_BOUNDS["min_samples_leaf"],
                )
            ),
            "max_features": _clamp(
                current_config.get("max_features", 0.8) * (1.1 if step % 2 == 0 else 0.9),
                PARAM_BOUNDS["max_features"],
            ),
            "bootstrap": current_config.get("bootstrap", False),
        }

    def sanitize_config(self, proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        return sanitize_with_clamp_tracking(
            proposal, PARAM_BOUNDS,
            integer_keys={"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap"},
        )

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback (lower is better for RMSE)."""
        return metrics.get("rmse", 0.0)

    def _is_higher_better(self) -> bool:
        """RMSE: lower is better."""
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
        return "You are tuning an ExtraTreesRegressor for California housing price prediction."

    def _get_output_format_instructions(self) -> str:
        """Return output format instructions."""
        return "Values must be numeric and within reasonable ranges."


def run_housing_bench(args, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run Housing benchmark."""
    config = BenchmarkConfig.from_args(args)
    benchmark = HousingBenchmark(config)
    return benchmark.run(run_id=run_id or args.run_id)
