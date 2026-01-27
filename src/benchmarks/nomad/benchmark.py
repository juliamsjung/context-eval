"""NOMAD benchmark implementation."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, _clamp
from src.benchmarks.nomad.env import NomadEnv


DEFAULT_HISTORY_WINDOW = 5

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

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, project_config)
        self.env = NomadEnv()
        self.context_summary = self.env.read_context()

    @property
    def benchmark_name(self) -> str:
        return "nomad"

    @property
    def dataset_id(self) -> str:
        return "nomad"

    @property
    def agent_id(self) -> str:
        return "nomad_llm"

    def get_default_config(self) -> Dict[str, Any]:
        return self.env.read_config()

    def run_training(self, config: Dict[str, Any]) -> Dict[str, float]:
        self.env.write_config(config)
        results = self.env.run_train()
        return {
            "mae": results.get("metric_value", results.get("metrics", {}).get("mae", 0.0)),
            "rmse": results.get("metrics", {}).get("rmse", 0.0),
            "r2": results.get("metrics", {}).get("r2", 0.0),
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
                current_config.get("l2_regularization", 0.0) * (1.3 if step % 2 == 0 else 0.7),
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

    def sanitize_config(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, (low, high) in PARAM_BOUNDS.items():
            if key not in proposal:
                continue
            try:
                val = float(proposal[key])
                if key in {"max_depth", "max_leaf_nodes", "max_iter", "min_samples_leaf"}:
                    sanitized[key] = int(_clamp(round(val), (low, high)))
                else:
                    sanitized[key] = _clamp(val, (low, high))
            except (ValueError, TypeError):
                continue
        return sanitized

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback (lower is better for MAE)."""
        return metrics.get("mae", 0.0)

    def _get_llm_system_prompt(self) -> str:
        return "You are an ML assistant optimizing gradient boosting hyperparameters."

    def _build_llm_user_prompt(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> str:
        history_payload = [
            {
                "step": entry.step,
                "config": entry.config,
                "score": self._get_primary_score(entry.metrics),
            }
            for entry in history[-self.config.history_window:]
        ]

        prompt_payload = {
            "current_config": {k: current_config.get(k) for k in PARAM_BOUNDS.keys() if k in current_config},
            "latest_score": self._get_primary_score(last_metrics),
            "recent_history": history_payload,
        }
        # dataset_context removed - will be reintroduced via explicit visibility flags

        return (
            "You are tuning a HistGradientBoostingRegressor for a regression task. "
            "Use the structured information below to recommend "
            "a new configuration that improves the score.\n"
            f"{json.dumps(prompt_payload, indent=2)}\n\n"
            "Return JSON with numeric keys among "
            f"{list(PARAM_BOUNDS.keys())}. Keep values within reasonable ranges."
        )


def run_nomad_bench(
    num_steps: int = 3,
    *,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run NOMAD benchmark. Thin wrapper around NomadBenchmark."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        history_window=history_window,
        seed=seed,
    )
    benchmark = NomadBenchmark(bench_config, config or {})
    return benchmark.run(run_id=run_id)
