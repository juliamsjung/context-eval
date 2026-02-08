"""Toy tabular benchmark implementation with agent support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, _clamp, _validate_dict_keys_no_trace_fields
from src.benchmarks.toy.env import ToyTabularEnv
# CONTEXT ONLY import
from src.context import ContextBundle


PARAM_BOUNDS = {
    "C": (0.01, 100.0),
    "max_iter": (10, 1000),
}


class ToyTabularBenchmark(BaseBenchmark):
    """Toy logistic regression benchmark with agent support."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.env = ToyTabularEnv()

    @property
    def benchmark_name(self) -> str:
        return "toy"

    @property
    def dataset_id(self) -> str:
        return "toy"

    @property
    def agent_id(self) -> str:
        return "toy_llm"

    @property
    def workspace_path(self) -> Path:
        return Path(__file__).resolve().parent / "workspace"

    def get_default_config(self) -> Dict[str, Any]:
        return self.env.read_config()

    def run_training(self, config: Dict[str, Any]) -> Dict[str, float]:
        self.env.write_config(config)
        results = self.env.run_train()
        return {"accuracy": results["accuracy"]}

    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Deterministic heuristic if LLM proposals are unavailable."""
        factor = 1.4 if step % 2 == 0 else 0.8
        new_C = _clamp(current_config.get("C", 1.0) * factor, PARAM_BOUNDS["C"])
        new_iter = int(_clamp(current_config.get("max_iter", 100) + 50, PARAM_BOUNDS["max_iter"]))
        return {"C": round(new_C, 4), "max_iter": new_iter}

    def sanitize_config(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        if "C" in proposal:
            try:
                sanitized["C"] = _clamp(float(proposal["C"]), PARAM_BOUNDS["C"])
            except (ValueError, TypeError):
                pass
        if "max_iter" in proposal:
            try:
                sanitized["max_iter"] = int(_clamp(int(proposal["max_iter"]), PARAM_BOUNDS["max_iter"]))
            except (ValueError, TypeError):
                pass
        return sanitized

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback."""
        return metrics.get("accuracy", 0.0)

    def _get_llm_system_prompt(self) -> str:
        return "You propose new logistic regression hyperparameters based on past evaluations."

    def _build_llm_user_prompt(
        self,
        bundle: ContextBundle,
    ) -> str:
        """
        CONTEXT ONLY: Build the user prompt from a validated ContextBundle.

        Args:
            bundle: Validated ContextBundle containing only agent-visible data
        """
        # Filter config to only tunable params (already validated by bundle)
        filtered_config = {
            'C': bundle.current_config.get('C'),
            'max_iter': bundle.current_config.get('max_iter')
        }

        # Validate structures before serialization (checks keys, not values)
        if __debug__:
            _validate_dict_keys_no_trace_fields(filtered_config)
            if bundle.resource_summary:
                _validate_dict_keys_no_trace_fields(bundle.resource_summary)

        # Format history as text lines
        if bundle.recent_history:
            history_lines = "\n".join(
                f"- step {e['step']}: score={e['score']:.4f}, C={e['config'].get('C')}, max_iter={e['config'].get('max_iter')}"
                for e in bundle.recent_history
            )
        else:
            history_lines = "- baseline only"

        prompt = "You are adjusting hyperparameters for logistic regression.\n\n"
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

        prompt += "Return JSON with numeric keys 'C' and 'max_iter'. Keep values positive and reasonable."
        return prompt


def run_toy_tabular(
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
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Toy benchmark."""
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
        verbose=verbose,
    )
    benchmark = ToyTabularBenchmark(bench_config)
    result = benchmark.run(run_id=run_id)

    # Convert to legacy format for compatibility
    return {
        "final_accuracy": result["final_metrics"].get("accuracy"),
        "final_config": result["final_config"],
        "num_steps": num_steps,
        "history": result["history"],
    }
