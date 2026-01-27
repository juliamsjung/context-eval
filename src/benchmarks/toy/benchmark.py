"""Toy tabular benchmark implementation with agent support."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, _clamp
from src.benchmarks.toy.env import ToyTabularEnv


PARAM_BOUNDS = {
    "C": (0.01, 100.0),
    "max_iter": (10, 1000),
}


class ToyTabularBenchmark(BaseBenchmark):
    """Toy logistic regression benchmark with agent support."""

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, project_config)
        self.env = ToyTabularEnv()

    @property
    def benchmark_name(self) -> str:
        return "toy_tabular"

    @property
    def dataset_id(self) -> str:
        return "toy_tabular"

    @property
    def agent_id(self) -> str:
        return "toy_llm"

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

    def _get_llm_system_prompt(self) -> str:
        return "You propose new logistic regression hyperparameters based on past evaluations."

    def _build_llm_user_prompt(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> str:
        history_lines = "\n".join(
            f"- step {entry.step}: accuracy={entry.metrics.get('accuracy', 0):.4f}, "
            f"C={entry.config.get('C')}, max_iter={entry.config.get('max_iter')}"
            for entry in history
        )
        if not history_lines:
            history_lines = "- baseline only"

        return (
            "You are adjusting hyperparameters for logistic regression on a fixed synthetic dataset.\n"
            f"Current config:\n{json.dumps({'C': current_config.get('C'), 'max_iter': current_config.get('max_iter')}, indent=2)}\n\n"
            f"Latest metrics:\n{json.dumps(last_metrics, indent=2)}\n\n"
            f"History:\n{history_lines}\n\n"
            "Return JSON with numeric keys 'C' and 'max_iter'. Keep values positive and reasonable."
        )


def run_toy_tabular(
    num_steps: int = 3,
    *,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Toy benchmark."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        seed=seed,
    )
    benchmark = ToyTabularBenchmark(bench_config, config or {})
    result = benchmark.run(run_id=run_id)

    # Convert to legacy format for compatibility
    return {
        "final_accuracy": result["final_metrics"].get("accuracy"),
        "final_config": result["final_config"],
        "num_steps": num_steps,
        "history": result["history"],
    }
