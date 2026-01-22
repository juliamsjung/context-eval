"""Toy tabular benchmark implementation with agent support."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from src.utils.config import get_env_var
from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, safe_parse_json
from src.benchmarks.toy.env import ToyTabularEnv
from src.agent import AgentRunner, build_toy_tools, create_policy

logger = logging.getLogger(__name__)

PARAM_BOUNDS = {
    "C": (0.01, 100.0),
    "max_iter": (10, 1000),
}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


class ToyTabularBenchmark(BaseBenchmark):
    """Toy logistic regression benchmark with agent support."""

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, project_config)
        self.env = ToyTabularEnv()
        self.agent_runner: Optional[AgentRunner] = None
        self.policy_obj = None
        self._setup_agent_if_needed()

    def _setup_agent_if_needed(self) -> None:
        """Initialize AgentRunner if reasoning_mode is 'agentic'."""
        if self.config.reasoning_mode != "agentic":
            return

        context_policies = self.project_config.get("context_policies", {})
        policy_overrides = context_policies.get(self.config.policy_type, {})
        self.policy_obj = create_policy(self.config.policy_type, policy_overrides)

        agent_cfg = self.project_config.get("agentic", {})
        agent_cfg.setdefault("dataset_id", "toy_tabular")
        agent_cfg.setdefault("agent_id", self.project_config.get("project_name", "toy_agent"))

        context_summary = self._build_context_summary()
        tools = build_toy_tools(
            context_summary=context_summary,
            retrieval_config=self.policy_obj.config,
            clarifier_defaults={
                "what metric should i optimize?": "Accuracy - higher is better.",
                "what are the parameter bounds?": f"C: {PARAM_BOUNDS['C']}, max_iter: {PARAM_BOUNDS['max_iter']}",
            },
        )
        self.agent_runner = AgentRunner(
            policy=self.policy_obj,
            tools=tools,
            logger=None,
            config=agent_cfg,
        )

    def _build_context_summary(self) -> Dict[str, Any]:
        """Build context for agent about the toy task."""
        return {
            "task": "Tune LogisticRegression hyperparameters",
            "parameters": list(PARAM_BOUNDS.keys()),
            "bounds": PARAM_BOUNDS,
            "metric": "accuracy (higher is better)",
            "dataset": "1000 samples, 20 features, binary classification",
        }

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

    def propose_config(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, int]]]:
        """Propose config using agent (agentic) or direct LLM (controller)."""
        step = len(history)

        if self.config.reasoning_mode == "agentic" and self.agent_runner:
            return self._agent_propose(current_config, last_metrics, history, step)

        # Controller mode: direct LLM call
        return self._llm_propose(current_config, last_metrics, history)

    def _agent_propose(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
        step: int,
    ) -> tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, int]]]:
        """Use ReAct-style agent to propose config."""
        history_dicts = [
            {
                "step": r.step,
                "config": {"C": r.config.get("C"), "max_iter": r.config.get("max_iter")},
                "metrics": r.metrics,
            }
            for r in history
        ]

        agent_state = {
            "context_excerpt": json.dumps(self._build_context_summary(), sort_keys=True),
            "history": history_dicts[-self.config.history_window:],
            "clarification_hints": {
                "what metric should i optimize?": "Accuracy - higher is better.",
            },
        }

        task_input = (
            f"Iteration {step}/{self.config.num_steps}: improve LogisticRegression "
            f"for the Toy tabular benchmark. Current accuracy={last_metrics.get('accuracy', 'N/A')}. "
            f"Propose new hyperparameters C and max_iter."
        )

        agent_result = self.agent_runner.run(
            task_input=task_input,
            state=agent_state,
            run_id=self.logger.run_id if self.logger else "unknown",
            task_id="toy_tabular",
            reasoning_mode=self.config.reasoning_mode,
            iteration_idx=step,
        )

        # Extract proposal from agent output
        proposal = self._extract_proposal_from_agent(agent_result)
        if proposal:
            source = f"agent:{self.policy_obj.name if self.policy_obj else self.config.policy_type}"
            return proposal, source, agent_result.token_usage

        # Fall back to LLM if agent didn't produce proposal
        proposal, usage = self._direct_llm_propose(current_config, last_metrics, history)
        if proposal:
            return proposal, "agent_fallback_llm", usage

        return None, "heuristic", None

    def _extract_proposal_from_agent(self, agent_result) -> Optional[Dict[str, Any]]:
        """Extract and sanitize proposal from agent result."""
        structured = agent_result.structured_output
        if "proposal" in structured and isinstance(structured["proposal"], dict):
            return self.sanitize_config(structured["proposal"])

        # Try parsing final answer
        answer = agent_result.final_answer
        if answer:
            parsed = safe_parse_json(answer)
            if parsed:
                if "proposal" in parsed:
                    return self.sanitize_config(parsed["proposal"])
                return self.sanitize_config(parsed)
        return None

    def _llm_propose(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, int]]]:
        """Controller mode: direct LLM call."""
        proposal, usage = self._direct_llm_propose(current_config, last_metrics, history)
        if proposal:
            return proposal, "llm", usage
        return None, "heuristic", None

    def _direct_llm_propose(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Make direct LLM call to propose config."""
        if not OPENAI_AVAILABLE:
            return None, None

        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            return None, None

        history_lines = "\n".join(
            f"- step {entry.step}: accuracy={entry.metrics.get('accuracy', 0):.4f}, "
            f"C={entry.config.get('C')}, max_iter={entry.config.get('max_iter')}"
            for entry in history
        )
        if not history_lines:
            history_lines = "- baseline only"

        user_prompt = (
            "You are adjusting hyperparameters for logistic regression on a fixed synthetic dataset.\n"
            f"Current config:\n{json.dumps({'C': current_config.get('C'), 'max_iter': current_config.get('max_iter')}, indent=2)}\n\n"
            f"Latest metrics:\n{json.dumps(last_metrics, indent=2)}\n\n"
            f"History:\n{history_lines}\n\n"
            "Return JSON with numeric keys 'C' and 'max_iter'. Keep values positive and reasonable."
        )

        client = OpenAI(api_key=api_key)
        t0 = datetime.utcnow().timestamp()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You propose new logistic regression hyperparameters based on past evaluations.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = datetime.utcnow().timestamp() - t0
        content = resp.choices[0].message.content or ""
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "latency_sec": latency,
        }

        parsed = safe_parse_json(content)
        if not parsed:
            logger.warning("LLM response was not valid JSON: %s", content)
            return None, usage

        sanitized = self.sanitize_config(parsed)
        return sanitized or None, usage

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
