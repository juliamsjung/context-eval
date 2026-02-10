"""Context visibility axes configuration.

CONTEXT ONLY: This module defines the visibility controls that determine
what information is included in agent-visible context bundles.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextAxes:
    """
    CONTEXT ONLY: Immutable configuration for context visibility axes.

    These axes control what information is exposed to the agent in its
    context bundle. They map to CLI flags and determine the agent's
    information access.

    Attributes:
        history_window: Number of past iterations to include (0 = none)
        show_task: Whether to include task_description.txt content
        show_metric: Whether to include metric_description.txt content
        show_resources: Whether to include resource usage (tokens, cost, latency)
    """
    history_window: int = 0
    show_task: bool = False
    show_metric: bool = False
    show_resources: bool = False

    def __post_init__(self) -> None:
        """Validate axis values."""
        if self.history_window < 0:
            raise ValueError(f"history_window must be >= 0, got {self.history_window}")
