"""Jigsaw Toxic Comment Classification benchmark implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, _clamp, _validate_dict_keys_no_trace_fields
from src.benchmarks.jigsaw.env import JigsawEnv
# CONTEXT ONLY import
from src.context import ContextBundle


PARAM_BOUNDS = {
    "max_features": (1000, 50000),
    "ngram_max": (1, 3),
    "min_df": (1, 20),
    "C": (0.01, 10.0),
    "max_iter": (50, 500),
}


class JigsawBenchmark(BaseBenchmark):
    """Jigsaw Toxic Comment multi-label text classification benchmark."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.env = JigsawEnv()

    @property
    def benchmark_name(self) -> str:
        return "jigsaw"

    @property
    def dataset_id(self) -> str:
        return "jigsaw"

    @property
    def agent_id(self) -> str:
        return "jigsaw_llm"

    @property
    def workspace_path(self) -> Path:
        return Path(__file__).resolve().parent / "workspace"

    def get_default_config(self) -> Dict[str, Any]:
        return self.env.read_config()

    def run_training(self, config: Dict[str, Any]) -> Dict[str, float]:
        self.env.write_config(config)
        results = self.env.run_train()
        return {
            "mean_auc": results.get("metric_value", results.get("metrics", {}).get("mean_auc", 0.0)),
            "num_labels_scored": results.get("metrics", {}).get("num_labels_scored", 0),
        }

    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        factor = 1.15 if step % 2 == 0 else 0.85
        return {
            "max_features": int(
                _clamp(
                    current_config.get("max_features", 10000) * factor,
                    PARAM_BOUNDS["max_features"],
                )
            ),
            "ngram_max": int(
                _clamp(
                    current_config.get("ngram_max", 2) + (1 if step % 3 == 0 else 0),
                    PARAM_BOUNDS["ngram_max"],
                )
            ),
            "min_df": int(
                _clamp(
                    current_config.get("min_df", 5) + (1 if step % 2 == 0 else -1),
                    PARAM_BOUNDS["min_df"],
                )
            ),
            "C": _clamp(
                current_config.get("C", 1.0) * factor,
                PARAM_BOUNDS["C"],
            ),
            "max_iter": int(
                _clamp(
                    current_config.get("max_iter", 100) + (25 if step % 2 == 0 else -15),
                    PARAM_BOUNDS["max_iter"],
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
                if key in {"max_features", "ngram_max", "min_df", "max_iter"}:
                    sanitized[key] = int(_clamp(round(val), (low, high)))
                else:
                    sanitized[key] = _clamp(val, (low, high))
            except (ValueError, TypeError):
                continue
        return sanitized

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback (higher is better for AUC)."""
        return metrics.get("mean_auc", 0.0)

    def _get_llm_system_prompt(self) -> str:
        return "You are an ML assistant optimizing text classification hyperparameters for toxicity detection."

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

        # Build dict representation for JSON serialization
        bundle_dict = {
            "current_config": filtered_config,
            "latest_score": bundle.latest_score,
            "recent_history": bundle.recent_history,
        }
        if bundle.task_description:
            bundle_dict["task_description"] = bundle.task_description
        if bundle.metric_description:
            bundle_dict["metric_description"] = bundle.metric_description
        if bundle.resource_summary:
            bundle_dict["resource_summary"] = bundle.resource_summary

        # Validate bundle_dict structure before serialization (checks keys, not values)
        if __debug__:
            _validate_dict_keys_no_trace_fields(bundle_dict)

        return (
            "You are tuning a TF-IDF + Logistic Regression pipeline for multi-label toxicity classification. "
            "Higher mean_auc is better (maximize). "
            "Use the structured information below to recommend "
            "a new configuration that improves the score.\n"
            f"{json.dumps(bundle_dict, indent=2)}\n\n"
            "Return JSON with numeric keys among "
            f"{list(PARAM_BOUNDS.keys())}. Keep values within reasonable ranges."
        )


def run_jigsaw_bench(
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
) -> Dict[str, Any]:
    """Run Jigsaw benchmark. Thin wrapper around JigsawBenchmark."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        history_window=history_window,
        seed=seed,
        show_task=show_task,
        show_metric=show_metric,
        show_resources=show_resources,
        model=model,
        temperature=temperature,
    )
    benchmark = JigsawBenchmark(bench_config)
    return benchmark.run(run_id=run_id)
