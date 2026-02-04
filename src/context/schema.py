"""Context layer schema definitions.

CONTEXT ONLY: This module defines the data structures for agent-visible context.
These structures are validated to ensure no trace-only data leaks into them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ContextLeakageError(Exception):
    """Raised when trace-only fields are detected in a context bundle.

    This exception indicates a structural violation: data intended only for
    observability/debugging has leaked into agent-visible context. This is
    a programming error and should be fixed immediately.
    """
    pass


@dataclass(frozen=True)
class ContextBundle:
    """
    CONTEXT ONLY: Immutable container for agent-visible context.

    This bundle contains only information that should be visible to the agent.
    Full metric dictionaries are logged for analysis and reproducibility but
    reduced to a scalar for agent context to preserve baseline invariance.

    Validation is structural: we check field names to ensure no trace-only
    fields (token_usage, api_cost, latency_sec, etc.) appear in the bundle.

    Attributes:
        current_config: The current hyperparameter configuration
        latest_score: Scalar score from the most recent evaluation
        recent_history: Windowed history of past iterations (step, config, score only)
        task_description: Optional task context (gated by --show-task)
        metric_description: Optional metric context (gated by --show-metric)
    """
    current_config: Dict[str, Any]
    latest_score: float
    recent_history: List[Dict[str, Any]] = field(default_factory=list)
    task_description: Optional[str] = None
    metric_description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate no trace-only fields leaked into the bundle."""
        self._validate_no_trace_leakage()

    def _validate_no_trace_leakage(self) -> None:
        """Structural check: reject if any trace-only field names are present."""
        from src.trace.schema import TRACE_ONLY_FIELDS

        # Check current_config
        for key in self.current_config:
            if key in TRACE_ONLY_FIELDS:
                raise ContextLeakageError(
                    f"ContextBundle must not contain trace-only fields. "
                    f"Found '{key}' in current_config."
                )

        # Check recent_history entries
        for i, entry in enumerate(self.recent_history):
            for key in entry:
                if key in TRACE_ONLY_FIELDS:
                    raise ContextLeakageError(
                        f"ContextBundle must not contain trace-only fields. "
                        f"Found '{key}' in recent_history[{i}]."
                    )
            # Also check nested config in history entries
            if "config" in entry and isinstance(entry["config"], dict):
                for key in entry["config"]:
                    if key in TRACE_ONLY_FIELDS:
                        raise ContextLeakageError(
                            f"ContextBundle must not contain trace-only fields. "
                            f"Found '{key}' in recent_history[{i}].config."
                        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/prompt building."""
        result: Dict[str, Any] = {
            "current_config": self.current_config,
            "latest_score": self.latest_score,
            "recent_history": self.recent_history,
        }
        if self.task_description is not None:
            result["task_description"] = self.task_description
        if self.metric_description is not None:
            result["metric_description"] = self.metric_description
        return result
