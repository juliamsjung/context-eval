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
        self._setup_agent_if_needed()

    @property
    def benchmark_name(self) -> str:
        return "toy_tabular"

    @property
    def dataset_id(self) -> str:
        return "toy_tabular"

    @property
    def agent_id(self) -> str:
        return f"toy_{self.config.reasoning_mode}_{self.config.policy_type}"

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

    # Abstract hook implementations
    def _build_agent_tools(self) -> Any:
        """Build tools for the toy benchmark agent."""
        from src.agent import build_toy_tools

        context_summary = self._build_context_summary()
        return build_toy_tools(
            context_summary=context_summary,
            retrieval_config=self.policy_obj.config,
            clarifier_defaults={
                "what metric should i optimize?": "Accuracy - higher is better.",
                "what are the parameter bounds?": f"C: {PARAM_BOUNDS['C']}, max_iter: {PARAM_BOUNDS['max_iter']}",
            },
        )

    def _get_agent_config_defaults(self) -> Dict[str, str]:
        return {
            "dataset_id": "toy_tabular",
            "agent_id": self.project_config.get("project_name", "toy_agent"),
        }

    def _build_context_summary(self) -> Dict[str, Any]:
        """Build context for agent about the toy task."""
        return {
            "task": "Tune LogisticRegression hyperparameters",
            "parameters": list(PARAM_BOUNDS.keys()),
            "bounds": PARAM_BOUNDS,
            "metric": "accuracy (higher is better)",
            "dataset": "1000 samples, 20 features, binary classification",
        }

    def _build_agent_state(self, history: List[IterationResult]) -> Dict[str, Any]:
        history_dicts = [
            {
                "step": r.step,
                "config": {"C": r.config.get("C"), "max_iter": r.config.get("max_iter")},
                "metrics": r.metrics,
            }
            for r in history
        ]
        return {
            "context_excerpt": json.dumps(self._build_context_summary(), sort_keys=True),
            "history": history_dicts[-self.config.history_window:],
            "clarification_hints": {
                "what metric should i optimize?": "Accuracy - higher is better.",
            },
        }

    def _build_agent_task_input(self, step: int, last_metrics: Dict[str, float]) -> str:
        return (
            f"Iteration {step}/{self.config.num_steps}: improve LogisticRegression "
            f"for the Toy tabular benchmark. Current accuracy={last_metrics.get('accuracy', 'N/A')}. "
            f"Propose new hyperparameters C and max_iter."
        )

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
    policy_type: str = "short_context",
    reasoning_mode: str = "controller",
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Toy benchmark with agent support."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        policy_type=policy_type,
        reasoning_mode=reasoning_mode,
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
