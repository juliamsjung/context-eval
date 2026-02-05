"""Abstract base class for benchmarks."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from src.utils.config import get_env_var
# TRACE ONLY imports
from src.trace import RunLogger, start_run, TRACE_ONLY_FIELDS
# CONTEXT ONLY imports
from src.context import ContextBundle, ContextAxes, ContextBuilder


# Model pricing per million tokens (as of Jan 2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}
DEFAULT_MODEL = "gpt-4o-mini"


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    """Clamp a value to the given bounds."""
    low, high = bounds
    return max(low, min(high, value))


class BaseEnv(ABC):
    """Abstract base class for benchmark environments."""

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or self._get_default_workspace()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.base_config_path = self.workspace / "config.json"
        self.config_path = self.workspace / "run_config.json"
        self.results_path = self.workspace / "results.json"
        self.train_script = self.workspace / "train.py"
        self._init_config()

    @abstractmethod
    def _get_default_workspace(self) -> Path:
        """Return the default workspace path for this environment."""
        ...

    def _init_config(self) -> None:
        """Initialize run_config.json from base config (always reset to ensure clean baseline)."""
        if self.base_config_path.exists():
            self.config_path.write_text(self.base_config_path.read_text())

    def read_config(self) -> Dict[str, Any]:
        """Read the current run configuration."""
        return json.loads(self.config_path.read_text())

    def write_config(self, cfg: Dict[str, Any]) -> None:
        """Write a new run configuration."""
        self.config_path.write_text(json.dumps(cfg, indent=2))

    def run_train(self) -> Dict[str, Any]:
        """Execute the training script and return results."""
        subprocess.run(
            [sys.executable, str(self.train_script)],
            cwd=self.workspace,
            check=True,
        )
        return json.loads(self.results_path.read_text())


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    num_steps: int = 3
    history_window: int = 5
    seed: int = 0
    show_task: bool = False
    show_metric: bool = False
    show_resources: bool = False


@dataclass
class IterationResult:
    """Result of a single benchmark iteration."""
    step: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    proposal_source: str  # "llm", "agent", "heuristic", "baseline"
    token_usage: Optional[Dict[str, int]] = None


class BaseBenchmark(ABC):
    """Abstract base class for ML experimentation benchmarks.

    This class manages the boundary between TRACE and CONTEXT layers:
    - TRACE: RunLogger handles all observability/debugging (token usage, costs, etc.)
    - CONTEXT: ContextBuilder constructs agent-visible information only
    """

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.project_config = project_config or {}
        self.history: List[IterationResult] = []
        # TRACE ONLY: Logger for observability events
        self.logger: Optional[RunLogger] = None
        # CONTEXT ONLY: Builder for agent-visible context bundles
        self._context_axes = ContextAxes(
            history_window=config.history_window,
            show_task=config.show_task,
            show_metric=config.show_metric,
            show_resources=config.show_resources,
        )
        # Note: _context_builder is initialized lazily after workspace_path is available
        self._context_builder: Optional[ContextBuilder] = None

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Unique identifier for this benchmark (e.g., 'nomad', 'toy_tabular')."""
        ...

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Dataset identifier for logging."""
        ...

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Agent identifier for logging."""
        ...

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return default hyperparameter configuration."""
        ...

    @abstractmethod
    def run_training(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run training with given config, return metrics dict."""
        ...

    @abstractmethod
    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Generate fallback config when LLM/agent fails."""
        ...

    @abstractmethod
    def sanitize_config(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp proposed config to valid bounds."""
        ...

    @abstractmethod
    def _get_llm_system_prompt(self) -> str:
        """Return the system prompt for direct LLM calls."""
        ...

    @abstractmethod
    def _build_llm_user_prompt(
        self,
        bundle: ContextBundle,
    ) -> str:
        """
        CONTEXT ONLY: Build the user prompt for direct LLM calls.

        Args:
            bundle: Validated ContextBundle containing only agent-visible data

        Returns:
            Formatted prompt string for the LLM
        """
        ...

    @property
    def workspace_path(self) -> Path:
        """Return workspace directory. Subclasses may override."""
        raise NotImplementedError("Subclass must define workspace_path")

    def _get_context_builder(self) -> ContextBuilder:
        """
        CONTEXT ONLY: Get or create the context builder instance.

        Lazily initializes the builder to allow workspace_path to be defined
        by subclasses after __init__.
        """
        if self._context_builder is None:
            try:
                workspace = self.workspace_path
            except NotImplementedError:
                workspace = None
            self._context_builder = ContextBuilder(
                axes=self._context_axes,
                score_extractor=self._get_primary_score,
                workspace_path=workspace,
            )
        return self._context_builder

    def _build_context_bundle(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> ContextBundle:
        """
        CONTEXT ONLY: Build validated context bundle for agent consumption.

        Delegates to ContextBuilder which enforces the trace/context boundary.
        Full metric dictionaries are logged for analysis and reproducibility but
        reduced to a scalar for agent context to preserve baseline invariance.

        Returns:
            Validated ContextBundle (raises ContextLeakageError if trace fields detected)
        """
        builder = self._get_context_builder()
        return builder.build(current_config, last_metrics, history)

    @abstractmethod
    def _get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract primary score for agent feedback. Subclass must implement."""
        ...

    def propose_config(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, int]]]:
        """Propose config using direct LLM call."""
        return self._llm_propose(current_config, last_metrics, history)

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

        # CONTEXT ONLY: Build validated context bundle for agent
        bundle = self._build_context_bundle(current_config, last_metrics, history)

        system_prompt = self._get_llm_system_prompt()
        user_prompt = self._build_llm_user_prompt(bundle)

        # Debug assertion: verify no trace fields leaked into prompt
        if __debug__:
            for field in TRACE_ONLY_FIELDS:
                assert field not in user_prompt, f"Trace field '{field}' leaked to prompt"

        client = OpenAI(api_key=api_key)
        t0 = datetime.utcnow().timestamp()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=250,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = datetime.utcnow().timestamp() - t0
        content = resp.choices[0].message.content or ""

        # TRACE LAYER: Compute cost per call using pricing formulas
        model = self.project_config.get("model", DEFAULT_MODEL)
        pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        api_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": resp.usage.total_tokens,
            "latency_sec": latency,
            "api_cost": round(api_cost, 8),  # Store cost per call for context aggregation
        }

        parsed = safe_parse_json(content)
        if not parsed:
            return None, usage

        sanitized = self.sanitize_config(parsed)
        return sanitized or None, usage

    def setup_logger(self, run_id: Optional[str] = None) -> RunLogger:
        """Initialize the run logger."""
        self.logger = start_run(
            task_id=self.benchmark_name,
            dataset_id=self.dataset_id,
            agent_id=self.agent_id,
            run_id=run_id,
        )
        return self.logger

    def _compute_run_totals(self) -> Dict[str, Any]:
        """Compute aggregate token/cost/latency metrics from history."""
        total_input = 0
        total_output = 0
        total_latency = 0.0

        for result in self.history:
            if result.token_usage:
                total_input += result.token_usage.get("input_tokens", 0)
                total_output += result.token_usage.get("output_tokens", 0)
                total_latency += result.token_usage.get("latency_sec", 0.0)

        total_tokens = total_input + total_output

        # Calculate cost (use model from config or default)
        model = self.project_config.get("model", DEFAULT_MODEL)
        pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
        total_cost = (total_input * pricing["input"] + total_output * pricing["output"]) / 1_000_000

        return {
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_latency_sec": total_latency,
            "total_api_cost": round(total_cost, 6),
        }

    def run(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full benchmark run.

        Returns dict with:
        - history: List of IterationResult as dicts
        - final_config: Best config found
        - final_metrics: Metrics from final iteration
        - total_steps: Number of iterations completed
        """
        self.setup_logger(run_id)
        self.history = []

        # Get initial config and run baseline
        current_config = self.get_default_config()

        self.logger.log_run_start(
            config_hash=self._config_hash(current_config),
            max_steps=self.config.num_steps,
            seed=self.config.seed,
            experiment_tags={
                "history_window": self.config.history_window,
                "show_task": self.config.show_task,
                "show_metric": self.config.show_metric,
                "show_resources": self.config.show_resources,
            },
        )

        # Step 0: Baseline
        print(f"===> Baseline run ({self.benchmark_name})")
        baseline_metrics = self.run_training(current_config)
        self.history.append(IterationResult(
            step=0,
            config=current_config.copy(),
            metrics=baseline_metrics,
            proposal_source="baseline",
        ))
        self.logger.log_op("op.train", step_idx=0, details={
            "config": current_config,
            "metrics": baseline_metrics,
            "source": "baseline",
        })

        # Iteration loop
        for step in range(1, self.config.num_steps + 1):
            print(f"\n===> {self.benchmark_name} Step {step}/{self.config.num_steps}")

            # Propose new config
            proposal, source, token_usage = self.propose_config(
                current_config,
                self.history[-1].metrics,
                self.history,
            )

            if proposal is None:
                proposal = self.fallback_config(current_config, step)
                source = "heuristic"
                token_usage = None

            # Sanitize and apply
            proposal = self.sanitize_config(proposal)
            current_config.update(proposal)

            print(f"{source.upper()} proposal: {proposal}")

            self.logger.log_op("op.config_proposal", step_idx=step, details={
                "proposal": proposal,
                "source": source,
                "token_usage": token_usage,
            })

            # Run training
            metrics = self.run_training(current_config)

            # Record result
            result = IterationResult(
                step=step,
                config=current_config.copy(),
                metrics=metrics,
                proposal_source=source,
                token_usage=token_usage,
            )
            self.history.append(result)

            self.logger.log_op("op.train", step_idx=step, details={
                "config": current_config,
                "metrics": metrics,
                "source": source,
            })
            self.logger.log_step_summary(step_idx=step, details={
                "config": current_config,
                "metrics": metrics,
                "source": source,
            })

            print(f"Result: {json.dumps(metrics, indent=2)}")

        # Finalize
        run_totals = self._compute_run_totals()

        final_result = {
            "history": [self._result_to_dict(r) for r in self.history],
            "final_config": current_config,
            "final_metrics": self.history[-1].metrics,
            "total_steps": len(self.history),
            "task_metrics": run_totals,
        }

        self.logger.log_run_end(
            status="success",
            final_metric=list(self.history[-1].metrics.values())[0] if self.history[-1].metrics else None,
            best_step_idx=len(self.history) - 1,
            n_steps=self.config.num_steps,
            total_tokens=run_totals["total_tokens"],
            total_api_cost=run_totals["total_api_cost"],
            total_latency_sec=run_totals["total_latency_sec"],
            details=final_result,
        )

        print(f"\n===> Final {self.benchmark_name} summary")
        print(json.dumps(final_result, indent=2))

        return final_result

    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Generate short hash of config for logging."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha1(config_str.encode()).hexdigest()[:10]

    def _result_to_dict(self, result: IterationResult) -> Dict[str, Any]:
        """Convert IterationResult to dict for serialization."""
        return {
            "step": result.step,
            "config": result.config,
            "metrics": result.metrics,
            "proposal_source": result.proposal_source,
            "token_usage": result.token_usage,
        }


# Shared utilities

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Try to find JSON object directly
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
