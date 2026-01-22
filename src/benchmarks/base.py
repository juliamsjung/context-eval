"""Abstract base class for benchmarks."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib

from src.utils.logging import RunLogger, start_run


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    num_steps: int = 3
    policy_type: str = "short_context"
    reasoning_mode: str = "controller"
    history_window: int = 5
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationResult:
    """Result of a single benchmark iteration."""
    step: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    proposal_source: str  # "llm", "agent", "heuristic", "baseline"
    token_usage: Optional[Dict[str, int]] = None


class BaseBenchmark(ABC):
    """Abstract base class for ML experimentation benchmarks."""

    def __init__(self, config: BenchmarkConfig, project_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.project_config = project_config or {}
        self.history: List[IterationResult] = []
        self.logger: Optional[RunLogger] = None

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
    def propose_config(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResult],
    ) -> tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, int]]]:
        """
        Propose next config. Returns (proposal, source, token_usage).
        source is one of: "llm", "agent", "heuristic"
        Returns (None, "heuristic", None) if LLM fails, then call fallback_config.
        """
        ...

    @abstractmethod
    def fallback_config(self, current_config: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Generate fallback config when LLM/agent fails."""
        ...

    @abstractmethod
    def sanitize_config(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp proposed config to valid bounds."""
        ...

    def setup_logger(self, run_id: Optional[str] = None) -> RunLogger:
        """Initialize the run logger."""
        self.logger = start_run(
            task_id=self.benchmark_name,
            dataset_id=self.dataset_id,
            agent_id=self.agent_id,
            run_id=run_id,
        )
        return self.logger

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
        final_result = {
            "history": [self._result_to_dict(r) for r in self.history],
            "final_config": current_config,
            "final_metrics": self.history[-1].metrics,
            "total_steps": len(self.history),
        }

        self.logger.log_run_end(
            status="success",
            final_metric=list(self.history[-1].metrics.values())[0] if self.history[-1].metrics else None,
            best_step_idx=len(self.history) - 1,
            n_steps=self.config.num_steps,
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
