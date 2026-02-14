"""Toy tabular benchmark implementation with agent support."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.benchmarks.base import (
    BaseBenchmark, BenchmarkConfig, _clamp,
    sanitize_with_clamp_tracking,
)
from src.benchmarks.toy.env import ToyTabularEnv


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

    def sanitize_config(self, proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        return sanitize_with_clamp_tracking(
            proposal, PARAM_BOUNDS, integer_keys={"max_iter"}
        )

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback."""
        return metrics.get("accuracy", 0.0)

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for this benchmark."""
        return PARAM_BOUNDS

    def _get_task_intro(self) -> str:
        """Return task-specific introduction text."""
        return "You are adjusting hyperparameters for logistic regression."

    def _format_history_entry(self, entry: Dict[str, Any]) -> str:
        """Format history entry with hardcoded field names (preserves exact formatting)."""
        return f"- step {entry['step']}: score={entry['score']:.4f}, C={entry['config'].get('C')}, max_iter={entry['config'].get('max_iter')}"

    def _get_output_format_instructions(self) -> str:
        """Return output format instructions."""
        return "Values must be numeric."


def run_toy_tabular(args, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run Toy benchmark."""
    config = BenchmarkConfig.from_args(args)
    benchmark = ToyTabularBenchmark(config)
    result = benchmark.run(run_id=run_id or args.run_id)

    # Convert to legacy format for compatibility
    return {
        "final_accuracy": result["final_metrics"].get("accuracy"),
        "final_config": result["final_config"],
        "num_steps": args.num_steps,
        "history": result["history"],
    }
