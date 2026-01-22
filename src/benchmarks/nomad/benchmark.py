"""NOMAD benchmark implementation."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from src.utils.config import get_env_var
from src.benchmarks.base import BaseBenchmark, BenchmarkConfig, IterationResult, safe_parse_json
from src.benchmarks.nomad.env import NomadEnv
from src.agent import AgentRunner, build_nomad_tools, create_policy

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_WINDOW = 5
MAX_CONTEXT_ATTR_LENGTH = 512

PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.5),
    "max_depth": (2, 16),
    "max_iter": (50, 1000),
    "l2_regularization": (0.0, 2.0),
    "max_leaf_nodes": (15, 255),
    "min_samples_leaf": (5, 200),
}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


def _history_window(history: List[IterationResult], window: int) -> List[Dict[str, Any]]:
    """Convert recent history to payload for LLM."""
    recent = history[-window:]
    return [
        {
            "step": entry.step,
            "proposal_source": entry.proposal_source,
            "config": entry.config,
            "metrics": entry.metrics,
        }
        for entry in recent
    ]


def _clarification_hints_from_history(history: List[Dict[str, Any]]) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for entry in history:
        for clar in entry.get("clarifications", []) or []:
            q = str(clar.get("question") or "").strip().lower()
            ans = clar.get("answer") or clar.get("observation") or ""
            if q and ans:
                hints[q] = ans
    return hints


class NomadBenchmark(BaseBenchmark):
    """NOMAD materials science regression benchmark."""

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, project_config)
        self.env = NomadEnv()
        self.context_summary = self.env.read_context()
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
        agent_cfg.setdefault("dataset_id", "nomad")
        agent_cfg.setdefault("agent_id", self.project_config.get("project_name", "nomad_agent"))

        tools = build_nomad_tools(
            context_summary=self.context_summary,
            retrieval_config=self.policy_obj.config,
            clarifier_defaults=agent_cfg.get("clarifier_defaults"),
        )
        self.agent_runner = AgentRunner(
            policy=self.policy_obj,
            tools=tools,
            logger=None,  # No separate agent logger
            config=agent_cfg,
        )

    @property
    def benchmark_name(self) -> str:
        return "nomad"

    @property
    def dataset_id(self) -> str:
        return "nomad"

    @property
    def agent_id(self) -> str:
        return f"nomad_{self.config.reasoning_mode}_{self.config.policy_type}"

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
        # Build state for agent
        history_dicts = [
            {
                "step": r.step,
                "config": r.config,
                "metrics": r.metrics,
                "proposal_source": r.proposal_source,
            }
            for r in history
        ]

        agent_state = {
            "context_excerpt": json.dumps(self.context_summary, sort_keys=True)[:2000],
            "history": history_dicts[-self.config.history_window:],
            "clarification_hints": {
                "target metric": "mae",
                "dataset": "nomad",
            },
        }

        task_input = (
            f"Iteration {step}/{self.config.num_steps}: improve HistGradientBoostingRegressor "
            f"for the NOMAD benchmark. Current metric MAE={last_metrics.get('mae', 'N/A')}. "
            f"Propose new hyperparameters."
        )

        agent_result = self.agent_runner.run(
            task_input=task_input,
            state=agent_state,
            run_id=self.logger.run_id if self.logger else "unknown",
            task_id="nomad",
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

        history_payload = _history_window(history, self.config.history_window)

        prompt_payload = {
            "current_config": {k: current_config.get(k) for k in PARAM_BOUNDS.keys() if k in current_config},
            "latest_metrics": last_metrics,
            "recent_history": [
                {
                    "step": h["step"],
                    "config": h["config"],
                    "metrics": h["metrics"],
                }
                for h in history_payload
            ],
        }
        if self.context_summary:
            prompt_payload["dataset_context"] = self.context_summary

        user_prompt = (
            "You are tuning a HistGradientBoostingRegressor to predict bandgap energy (eV) "
            "from the NOMAD 2018 dataset. Use the structured information below to recommend "
            "a new configuration that lowers the evaluation metric (default: MAE).\n"
            f"{json.dumps(prompt_payload, indent=2)}\n\n"
            "Return JSON with numeric keys among "
            f"{list(PARAM_BOUNDS.keys())}. Keep values within reasonable ranges."
        )

        client = OpenAI(api_key=api_key)
        t0 = datetime.utcnow().timestamp()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=250,
            messages=[
                {
                    "role": "system",
                    "content": "You are an ML assistant optimizing gradient boosting hyperparameters.",
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


def run_nomad_bench(
    num_steps: int = 3,
    *,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    policy_type: str = "short_context",
    reasoning_mode: str = "controller",
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run NOMAD benchmark. Thin wrapper around NomadBenchmark."""
    bench_config = BenchmarkConfig(
        num_steps=num_steps,
        policy_type=policy_type,
        reasoning_mode=reasoning_mode,
        history_window=history_window,
        seed=seed,
    )
    benchmark = NomadBenchmark(bench_config, config or {})
    return benchmark.run(run_id=run_id)
