"""NOMAD benchmark implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, _clamp, _validate_dict_keys_no_trace_fields
from src.benchmarks.nomad.env import NomadEnv
# CONTEXT ONLY import
from src.context import ContextBundle


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
        bundle: ContextBundle,
    ) -> str:
        """
        CONTEXT ONLY: Build the user prompt from a validated ContextBundle.

        Args:
            bundle: Validated ContextBundle containing only agent-visible data
        """
        # Filter config to only tunable params (bundle already validated)
        filtered_config = {k: bundle.current_config.get(k) for k in PARAM_BOUNDS.keys() if k in bundle.current_config}

        # Validate structures before serialization (checks keys, not values)
        if __debug__:
            _validate_dict_keys_no_trace_fields(filtered_config)
            if bundle.resource_summary:
                _validate_dict_keys_no_trace_fields(bundle.resource_summary)

        # Format history as text lines
        if bundle.recent_history:
            history_lines = "\n".join(
                f"- step {e['step']}: score={e['score']:.4f}, "
                + ", ".join(f"{k}={v}" for k, v in e['config'].items())
                for e in bundle.recent_history
            )
        else:
            history_lines = "- baseline only"

        prompt = "You are tuning a HistGradientBoostingRegressor.\n\n"
        prompt += f"### Current Configuration\n{json.dumps(filtered_config, indent=2)}\n\n"
        prompt += f"### Latest Score\n{bundle.latest_score:.4f}\n\n"
        prompt += f"### History\n{history_lines}\n\n"

        # Add context sections if available (using markdown headers)
        if bundle.task_description:
            prompt += f"### Task Description\n{bundle.task_description}\n\n"
        if bundle.metric_description:
            prompt += f"### Evaluation Metric\n{bundle.metric_description}\n\n"
        if bundle.resource_summary:
            prompt += f"### Resource Usage\n{json.dumps(bundle.resource_summary, indent=2)}\n\n"

        prompt += (
            "Return JSON with numeric keys among "
            f"{list(PARAM_BOUNDS.keys())}. Keep values within reasonable ranges."
        )
        return prompt


def run_nomad_bench(
    num_steps: int,
    *,
    history_window: int,
    show_task: bool,
    show_metric: bool,
    show_resources: bool,
    seed: int,
    run_id: Optional[str],
    model: str,
    temperature: float,
    debug_show_prompt: bool = False,
) -> Dict[str, Any]:
    """Run NOMAD benchmark. Thin wrapper around NomadBenchmark."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        history_window=history_window,
        seed=seed,
        show_task=show_task,
        show_metric=show_metric,
        show_resources=show_resources,
        model=model,
        temperature=temperature,
        debug_show_prompt=debug_show_prompt,
    )
    benchmark = NomadBenchmark(bench_config)
    return benchmark.run(run_id=run_id)
