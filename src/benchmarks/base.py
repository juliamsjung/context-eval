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
from typing import Any, Dict, List, Optional, Tuple
import hashlib

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from src.utils.config import get_env_var
from src.config.paths import RUNS_ROOT
# TRACE ONLY imports
from src.trace import RunLogger, start_run, TRACE_ONLY_FIELDS
# CONTEXT ONLY imports
from src.context import ContextBundle, ContextAxes, ContextBuilder
# LOGGING LAYER imports
from src.logging import RunSummary


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


def _get_git_commit() -> Optional[str]:
    """Get short git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sanitize_with_clamp_tracking(
    proposal: Dict[str, Any],
    param_bounds: Dict[str, tuple[float, float]],
    integer_keys: set[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sanitize a config proposal by clamping values to bounds.

    Args:
        proposal: Raw config proposal from LLM
        param_bounds: Dict mapping param names to (low, high) bounds
        integer_keys: Set of param names that should be cast to int

    Returns:
        Tuple of (sanitized_config, clamp_events)
    """
    sanitized: Dict[str, Any] = {}
    clamp_events: List[Dict[str, Any]] = []

    for key, (low, high) in param_bounds.items():
        if key not in proposal:
            continue
        try:
            val = float(proposal[key])
            if key in integer_keys:
                # Round to int first, then clamp
                rounded = int(round(val))
                clamped = int(_clamp(rounded, (low, high)))
            else:
                clamped = _clamp(val, (low, high))
            sanitized[key] = clamped

            if val < low or val > high:
                clamp_events.append({
                    "parameter": key,
                    "proposed": val,
                    "executed": clamped,
                })
        except (ValueError, TypeError):
            continue

    return sanitized, clamp_events


def format_resources_section(bundle: ContextBundle) -> str:
    """Format the resources section for prompts."""
    if not bundle.resource_summary:
        return ""
    rs = bundle.resource_summary
    return (
        f"### Resources\n"
        f"tokens_current: {rs['tokens_current']}\n"
        f"tokens_cumulative: {rs['tokens_cumulative']}\n"
        f"cost_cumulative: {rs['cost_cumulative']}\n\n"
    )


def format_diagnostics_section(bundle: ContextBundle) -> str:
    """Format the diagnostics section for prompts."""
    if not bundle.diagnostics:
        return ""
    d = bundle.diagnostics
    lines = ["### Diagnostics\n"]
    clamp_events = d.get("clamp_events", [])
    if clamp_events:
        lines.append("clamp_events:\n")
        for ce in clamp_events:
            lines.append(
                f"  - parameter: {ce['parameter']}\n"
                f"    proposed: {ce['proposed']}\n"
                f"    executed: {ce['executed']}\n"
            )
    else:
        lines.append("clamp_events: []\n")
    lines.append(f"parse_failure: {str(d.get('parse_failure', False)).lower()}\n")
    lines.append(f"fallback_used: {str(d.get('fallback_used', False)).lower()}\n")
    lines.append(f"truncated: {str(d.get('truncated', False)).lower()}\n\n")
    return "".join(lines)


def format_context_sections(bundle: ContextBundle) -> str:
    """Format all optional context sections (task, metric, resources, diagnostics)."""
    sections = []
    if bundle.task_description:
        sections.append(f"### Task Description\n{bundle.task_description}\n\n")
    if bundle.metric_description:
        sections.append(f"### Evaluation Metric\n{bundle.metric_description}\n\n")
    sections.append(format_resources_section(bundle))
    sections.append(format_diagnostics_section(bundle))
    return "".join(sections)


def _validate_dict_keys_no_trace_fields(data: Any, path: str = "root") -> None:
    """
    Recursively validate that no trace-only field names appear as dictionary keys.
    
    This function checks dictionary keys (not values) to catch actual field leaks
    while allowing natural language text in values that may contain these words.
    
    Args:
        data: The data structure to validate (dict, list, or primitive)
        path: Current path in the structure for error messages
        
    Raises:
        AssertionError: If any trace-only field name appears as a dictionary key
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in TRACE_ONLY_FIELDS:
                raise AssertionError(
                    f"Trace field '{key}' leaked to prompt structure at path '{path}'"
                )
            # Recursively check nested structures
            _validate_dict_keys_no_trace_fields(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _validate_dict_keys_no_trace_fields(item, f"{path}[{i}]")


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
            capture_output=True,
        )
        return json.loads(self.results_path.read_text())


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    num_steps: int = 3
    feedback_depth: int = 1
    seed: int = 0
    show_task: bool = False
    show_metric: bool = False
    show_resources: bool = False
    show_diagnostics: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = 0
    experiment_id: str = "default"
    debug_show_llm: bool = False
    verbose: bool = False

    @classmethod
    def from_args(cls, args) -> "BenchmarkConfig":
        """Create config from argparse.Namespace."""
        return cls(
            num_steps=args.num_steps,
            feedback_depth=args.feedback_depth,
            seed=args.seed,
            show_task=args.show_task,
            show_metric=args.show_metric,
            show_resources=args.show_resources,
            show_diagnostics=args.show_diagnostics,
            model=args.model,
            temperature=args.temperature,
            experiment_id=args.experiment_id,
            debug_show_llm=args.debug_show_llm,
            verbose=args.verbose,
        )


@dataclass
class IterationResult:
    """Result of a single benchmark iteration."""
    step: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    proposal_source: str  # "llm", "heuristic", "baseline"
    token_usage: Optional[Dict[str, int]] = None
    diagnostics: Optional[Dict[str, Any]] = None


class BaseBenchmark(ABC):
    """Abstract base class for ML experimentation benchmarks.

    This class manages the boundary between TRACE and CONTEXT layers:
    - TRACE: RunLogger handles all observability/debugging (token usage, costs, etc.)
    - CONTEXT: ContextBuilder constructs agent-visible information only
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.history: List[IterationResult] = []
        # TRACE ONLY: Logger for observability events
        self.logger: Optional[RunLogger] = None
        # CONTEXT ONLY: Builder for agent-visible context bundles
        self._context_axes = ContextAxes(
            feedback_depth=config.feedback_depth,
            show_task=config.show_task,
            show_metric=config.show_metric,
            show_resources=config.show_resources,
            show_diagnostics=config.show_diagnostics,
        )
        # Note: _context_builder is initialized lazily after workspace_path is available
        self._context_builder: Optional[ContextBuilder] = None

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Unique identifier for this benchmark (e.g., 'nomad', 'toy')."""
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
    def sanitize_config(self, proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Validate and clamp proposed config to valid bounds.

        Returns:
            Tuple of (sanitized_config, clamp_events) where clamp_events is a list of
            dicts with 'parameter', 'proposed', and 'executed' keys for any values
            that were clamped.
        """
        ...

    def _get_llm_system_prompt(self) -> str:
        """Return the system prompt for direct LLM calls."""
        return (
            "You output ONLY valid JSON. "
            "No explanations, no markdown, no text outside the JSON object."
        )

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
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]], bool, List[Dict[str, Any]]]:
        """Propose config using direct LLM call.

        Returns:
            Tuple of (proposal, source, token_usage, parse_failure, clamp_events)
        """
        return self._llm_propose(current_config, last_metrics, history)

    def _llm_propose(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]], bool, List[Dict[str, Any]]]:
        """Controller mode: direct LLM call.

        Returns:
            Tuple of (proposal, source, token_usage, parse_failure, clamp_events)
        """
        proposal, usage, parse_failure, clamp_events = self._direct_llm_propose(current_config, last_metrics, history)
        if proposal:
            return proposal, "llm", usage, parse_failure, clamp_events
        return None, "heuristic", usage, parse_failure, clamp_events

    def _direct_llm_propose(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool, List[Dict[str, Any]]]:
        """Make direct LLM call to propose config.

        Returns:
            Tuple of (sanitized_config, usage, parse_failure, clamp_events)
        """
        if not OPENAI_AVAILABLE:
            return None, None, False, []

        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            return None, None, False, []

        # CONTEXT ONLY: Build validated context bundle for agent
        bundle = self._build_context_bundle(current_config, last_metrics, history)

        # Validate bundle structure before prompt building (checks keys, not values)
        if __debug__:
            bundle_dict = bundle.to_dict()
            _validate_dict_keys_no_trace_fields(bundle_dict)

        system_prompt = self._get_llm_system_prompt()
        user_prompt = self._build_llm_user_prompt(bundle)

        client = OpenAI(api_key=api_key)
        t0 = datetime.utcnow().timestamp()
        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=150,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = datetime.utcnow().timestamp() - t0
        content = resp.choices[0].message.content or ""

        # TRACE LAYER: Compute cost per call using pricing formulas
        pricing = MODEL_PRICING.get(self.config.model, MODEL_PRICING[DEFAULT_MODEL])
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        api_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": resp.usage.total_tokens,
            "latency_sec": latency,
            "api_cost": round(api_cost, 8),  # Store cost per call for context aggregation
            "finish_reason": resp.choices[0].finish_reason,
            "raw_llm_response": resp.model_dump(),
        }

        if self.config.debug_show_llm:
            print("\n[DEBUG] ==== LLM REQUEST BEGIN ====")
            print(f"System:\n{system_prompt}\n")
            print(f"User:\n{user_prompt}")
            print("[DEBUG] ==== LLM REQUEST END ====\n")

            print("[DEBUG] ==== LLM RESPONSE BEGIN ====")
            print(content)
            print("[DEBUG] ==== LLM RESPONSE END ====")
            print(f"[DEBUG] finish_reason: {resp.choices[0].finish_reason}, "
                  f"tokens: {input_tokens} in / {output_tokens} out\n")

        parsed = safe_parse_json(content)
        if not parsed:
            return None, usage, True, []

        sanitized, clamp_events = self.sanitize_config(parsed)
        return sanitized or None, usage, False, clamp_events

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

        # Calculate cost using configured model
        pricing = MODEL_PRICING.get(self.config.model, MODEL_PRICING[DEFAULT_MODEL])
        total_cost = (total_input * pricing["input"] + total_output * pricing["output"]) / 1_000_000

        return {
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_latency_sec": total_latency,
            "total_api_cost": round(total_cost, 6),
        }

    def _is_higher_better(self) -> bool:
        """Return True if higher scores are better. Subclasses can override."""
        return True  # Default: higher is better (e.g., accuracy, AUC)

    def _compute_run_summary(self, run_id: str) -> RunSummary:
        """Compute aggregated run summary for analysis."""
        scores = []
        total_input = total_output = 0
        total_latency = 0.0
        num_clamp_events = num_parse_failures = num_fallbacks = num_truncations = 0

        for r in self.history:
            scores.append(self._get_primary_score(r.metrics))
            if r.token_usage and isinstance(r.token_usage, dict):
                total_input += r.token_usage.get("input_tokens", 0)
                total_output += r.token_usage.get("output_tokens", 0)
                total_latency += r.token_usage.get("latency_sec", 0.0)
            if r.diagnostics:
                num_clamp_events += len(r.diagnostics.get("clamp_events", []))
                num_parse_failures += int(r.diagnostics.get("parse_failure", False))
                num_fallbacks += int(r.diagnostics.get("fallback_used", False))
                num_truncations += int(r.diagnostics.get("truncated", False))

        final_score = scores[-1] if scores else 0.0
        best_score = (max(scores) if self._is_higher_better() else min(scores)) if scores else 0.0
        total_tokens = total_input + total_output

        pricing = MODEL_PRICING.get(self.config.model, MODEL_PRICING[DEFAULT_MODEL])
        total_cost = (total_input * pricing["input"] + total_output * pricing["output"]) / 1_000_000

        # Compute axis signature for easy filtering
        axis_signature = (
            f"fd{self.config.feedback_depth}"
            f"_t{int(self.config.show_task)}"
            f"_m{int(self.config.show_metric)}"
            f"_r{int(self.config.show_resources)}"
            f"_d{int(self.config.show_diagnostics)}"
        )

        return RunSummary(
            benchmark=self.benchmark_name,
            seed=self.config.seed,
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            git_commit=_get_git_commit(),
            model_name=self.config.model,
            temperature=self.config.temperature,
            axis_signature=axis_signature,
            feedback_depth=self.config.feedback_depth,
            show_task=self.config.show_task,
            show_metric=self.config.show_metric,
            show_resources=self.config.show_resources,
            show_diagnostics=self.config.show_diagnostics,
            final_score=final_score,
            best_score=best_score,
            num_steps=self.config.num_steps,
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            num_clamp_events=num_clamp_events,
            num_parse_failures=num_parse_failures,
            num_fallbacks=num_fallbacks,
            num_truncations=num_truncations,
        )

    def _write_run_summary(self, summary: RunSummary) -> None:
        """Append run summary to JSONL file."""
        runs_dir = RUNS_ROOT / self.config.experiment_id
        runs_dir.mkdir(parents=True, exist_ok=True)

        output_path = runs_dir / f"{self.benchmark_name}_runs.jsonl"
        with open(output_path, "a") as f:
            f.write(json.dumps(summary.to_dict()) + "\n")

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
                "feedback_depth": self.config.feedback_depth,
                "show_task": self.config.show_task,
                "show_metric": self.config.show_metric,
                "show_resources": self.config.show_resources,
                "show_diagnostics": self.config.show_diagnostics,
                "model": self.config.model,
                "temperature": self.config.temperature,
            },
        )

        # Step 0: Baseline
        if self.config.verbose:
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
            if self.config.verbose:
                print(f"\n===> {self.benchmark_name} Step {step}/{self.config.num_steps}")

            # Propose new config
            proposal, source, token_usage, parse_failure, clamp_events = self.propose_config(
                current_config,
                self.history[-1].metrics,
                self.history,
            )

            fallback_used = False
            if proposal is None:
                fallback_proposal = self.fallback_config(current_config, step)
                proposal, clamp_events = self.sanitize_config(fallback_proposal)
                source = "heuristic"
                fallback_used = True

            current_config.update(proposal)

            # Compute diagnostics (execution layer)
            truncated = False
            if token_usage:
                truncated = token_usage.get("finish_reason") == "length"

            diagnostics = {
                "clamp_events": clamp_events,
                "parse_failure": parse_failure,
                "fallback_used": fallback_used,
                "truncated": truncated,
            }

            if self.config.verbose:
                print(f"{source.upper()} proposal: {proposal}")

            self.logger.log_op("op.config_proposal", step_idx=step, details={
                "proposal": proposal,
                "source": source,
                "token_usage": token_usage,
                "diagnostics": diagnostics,
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
                diagnostics=diagnostics,
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

            if self.config.verbose:
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
        )

        if self.config.verbose:
            print(f"\n===> Final {self.benchmark_name} summary")
            print(json.dumps(final_result, indent=2))

        # Always print final score
        primary_metric = self._get_primary_score(self.history[-1].metrics)
        print(f"Final score: {primary_metric:.4f}")

        # Write run summary (logging layer - post-hoc aggregation only)
        summary = self._compute_run_summary(run_id or self.logger.run_id)
        self._write_run_summary(summary)

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
