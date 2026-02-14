"""Toy tabular benchmark implementation with agent support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing import Tuple
from src.benchmarks.base import (
    BaseBenchmark, BenchmarkConfig, _clamp,
    sanitize_with_clamp_tracking, format_context_sections,
)
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

    def sanitize_config(self, proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        return sanitize_with_clamp_tracking(
            proposal, PARAM_BOUNDS, integer_keys={"max_iter"}
        )

    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback."""
        return metrics.get("accuracy", 0.0)

    def _build_llm_user_prompt(
        self,
        bundle: ContextBundle,
    ) -> str:
        """CONTEXT ONLY: Build the user prompt from a validated ContextBundle."""
        filtered_config = {k: bundle.current_config.get(k) for k in PARAM_BOUNDS.keys()}

        prompt = "### Task\nYou are adjusting hyperparameters for logistic regression.\n\n"
        prompt += f"### Current Configuration\n{json.dumps(filtered_config, indent=2)}\n\n"
        prompt += f"### Feedback\nscore: {bundle.latest_score:.4f}\n\n"

        if bundle.recent_history:
            history_lines = "\n".join(
                f"- step {e['step']}: score={e['score']:.4f}, C={e['config'].get('C')}, max_iter={e['config'].get('max_iter')}"
                for e in bundle.recent_history
            )
            prompt += f"### History\n{history_lines}\n\n"

        prompt += format_context_sections(bundle)
        prompt += (
            "### Output Format\n"
            f"Return JSON with exactly these keys: {list(PARAM_BOUNDS.keys())}.\n"
            "Values must be numeric."
        )
        return prompt


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
