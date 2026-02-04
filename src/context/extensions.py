"""Context extension interfaces for future features.

CONTEXT ONLY: This module defines extension interfaces for safely adding
new context features. Extensions allow injecting additional context into
bundles while maintaining the trace/context boundary.

Example future extensions:
- RationaleExtension: Add agent reasoning to context (CONTEXT)
- DeltaExtension: Add change explanations to context (CONTEXT)

Note: cost_so_far would be TRACE ONLY and should never be added via extension.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.context.schema import ContextBundle


class ContextExtension(ABC):
    """
    CONTEXT ONLY: Abstract base class for context extensions.

    Extensions allow adding new information to context bundles while
    respecting the trace/context boundary. Implementations must ensure
    they never add trace-only fields.

    Usage:
        class RationaleExtension(ContextExtension):
            def extend(self, bundle: ContextBundle) -> ContextBundle:
                # Add rationale to context
                return ContextBundle(
                    current_config=bundle.current_config,
                    latest_score=bundle.latest_score,
                    recent_history=bundle.recent_history,
                    task_description=bundle.task_description,
                    metric_description=bundle.metric_description,
                    # Future: rationale=self._generate_rationale(bundle),
                )
    """

    @abstractmethod
    def extend(self, bundle: ContextBundle) -> ContextBundle:
        """
        Extend a context bundle with additional information.

        Args:
            bundle: The base context bundle to extend

        Returns:
            A new ContextBundle with extended information

        Note:
            Implementations must not add trace-only fields.
            The returned bundle will be validated.
        """
        ...
