"""Context bundle builder.

CONTEXT ONLY: This module is responsible for constructing agent-visible
context bundles. It enforces the boundary between trace and context layers
by explicitly extracting only the data that should be visible to agents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol


from src.context.axes import ContextAxes
from src.context.schema import ContextBundle


class IterationResultProtocol(Protocol):
    """Protocol for iteration result objects."""
    step: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    token_usage: Optional[Dict[str, Any]]
    diagnostics: Optional[Dict[str, Any]]


class ContextBuilder:
    """
    CONTEXT ONLY: Builder for agent-visible context bundles.

    This class owns the construction of ContextBundle instances, ensuring
    that only appropriate data flows to the agent. Full metric dictionaries
    are logged for analysis and reproducibility but reduced to a scalar for
    agent context to preserve baseline invariance.

    The builder takes a score extractor function to convert full metrics
    dictionaries into scalar scores for the agent.

    Attributes:
        axes: Visibility configuration controlling what's included
        score_extractor: Function to extract scalar score from metrics dict
        workspace_path: Optional path to load artifact files from
    """

    def __init__(
        self,
        axes: ContextAxes,
        score_extractor: Callable[[Dict[str, float]], float],
        workspace_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            axes: Visibility axes configuration
            score_extractor: Function that extracts primary score from metrics
            workspace_path: Optional workspace path for loading artifacts
        """
        self.axes = axes
        self.score_extractor = score_extractor
        self.workspace_path = workspace_path

    def _load_artifact(self, filename: str) -> Optional[str]:
        """Load text artifact from workspace if it exists."""
        if self.workspace_path is None:
            return None
        path = self.workspace_path / filename
        if path.exists():
            return path.read_text().strip()
        return None

    def _get_task_description(self) -> Optional[str]:
        """Load task_description.txt if show_task is enabled."""
        if not self.axes.show_task:
            return None
        return self._load_artifact("task_description.txt")

    def _get_metric_description(self) -> Optional[str]:
        """Load metric_description.txt if show_metric is enabled."""
        if not self.axes.show_metric:
            return None
        return self._load_artifact("metric_description.txt")

    def _compute_resource_summary(
        self, history: List[IterationResultProtocol]
    ) -> Optional[Dict[str, Any]]:
        """
        CONTEXT ONLY: Aggregate resource usage from history.

        IMPORTANT: This method only SUMS existing values computed by the trace layer.
        It NEVER computes costs from pricing formulas - that's the trace layer's job.

        Args:
            history: Full iteration history with token_usage data

        Returns:
            Resource summary dict if show_resources is enabled, None otherwise
        """
        if not self.axes.show_resources:
            return None

        total_tokens = 0
        total_cost = 0.0
        last_tokens = 0

        for entry in history:
            if hasattr(entry, 'token_usage') and entry.token_usage and isinstance(entry.token_usage, dict):
                tokens = entry.token_usage.get("total_tokens", 0)
                total_tokens += tokens
                total_cost += entry.token_usage.get("api_cost", 0.0)  # Sum pre-computed cost
                last_tokens = tokens  # Only update when we have valid token usage

        return {
            "tokens_current": last_tokens,
            "tokens_cumulative": total_tokens,
            "cost_cumulative": round(total_cost, 6),
        }

    def _get_diagnostics(
        self, history: List[IterationResultProtocol]
    ) -> Optional[Dict[str, Any]]:
        """
        CONTEXT ONLY: Project diagnostics from most recent step.

        This method only READS pre-computed diagnostics from IterationResult.
        It NEVER computes truncation, parse_failure, etc. — that's the execution layer's job.

        Args:
            history: Full iteration history with diagnostics data

        Returns:
            Diagnostics dict if show_diagnostics is enabled, None otherwise
        """
        if not self.axes.show_diagnostics:
            return None

        if not history:
            return None

        last = history[-1]
        if not hasattr(last, 'diagnostics') or not last.diagnostics:
            return None

        return last.diagnostics

    def build(
        self,
        current_config: Dict[str, Any],
        last_metrics: Dict[str, float],
        history: List[IterationResultProtocol],
    ) -> ContextBundle:
        """
        Build a validated context bundle for agent consumption.

        This method constructs a ContextBundle by:
        1. Extracting scalar score from full metrics (trace-only data excluded)
        2. Windowing history according to visibility axes
        3. Optionally loading task/metric descriptions
        4. Validating no trace fields leaked through

        Args:
            current_config: Current hyperparameter configuration
            last_metrics: Full metrics dict from last evaluation (scalar extracted)
            history: Full iteration history (windowed and filtered for agent)

        Returns:
            Validated ContextBundle instance

        Raises:
            ContextLeakageError: If trace-only fields are detected
        """
        # Extract scalar score from full metrics
        latest_score = self.score_extractor(last_metrics)

        # Build windowed history with only agent-visible fields
        # Exclude last entry (current step) since latest_score provides it
        recent_history: List[Dict[str, Any]] = []
        if self.axes.feedback_depth > 1 and len(history) > 1:
            num_prev = min(self.axes.feedback_depth - 1, len(history) - 1)
            prev_entries = history[-num_prev - 1:-1]
            for entry in prev_entries:
                recent_history.append({
                    "step": entry.step,
                    "config": entry.config,
                    "score": self.score_extractor(entry.metrics),
                })

        # Load optional descriptions based on visibility flags
        task_desc = self._get_task_description()
        metric_desc = self._get_metric_description()

        # Compute resource summary if enabled (sums pre-computed costs, no pricing formulas)
        resource_summary = self._compute_resource_summary(history)

        # Get diagnostics if enabled (projects pre-computed diagnostics from last step)
        diagnostics = self._get_diagnostics(history)

        # Construct and validate bundle (validation happens in __post_init__)
        return ContextBundle(
            current_config=current_config,
            latest_score=latest_score,
            recent_history=recent_history,
            task_description=task_desc,
            metric_description=metric_desc,
            resource_summary=resource_summary,
            diagnostics=diagnostics,
        )
