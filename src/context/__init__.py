"""Context layer package for agent-visible information.

CONTEXT ONLY: This package handles the construction and validation of
agent-visible context bundles. It enforces hard boundaries to ensure
trace-only data never leaks into agent prompts.

Exports:
    - ContextBundle: Immutable container for agent-visible context
    - ContextAxes: Visibility axis configuration
    - ContextBuilder: Builder for constructing validated context bundles
    - ContextLeakageError: Exception raised when trace data leaks
    - ContextExtension: Abstract base class for context extensions
"""
from __future__ import annotations

from src.context.schema import ContextBundle, ContextLeakageError
from src.context.axes import ContextAxes
from src.context.builder import ContextBuilder
from src.context.extensions import ContextExtension

__all__ = [
    "ContextBundle",
    "ContextAxes",
    "ContextBuilder",
    "ContextLeakageError",
    "ContextExtension",
]
