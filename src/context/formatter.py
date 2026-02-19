"""Context formatting utilities for prompt construction.

CONTEXT ONLY: This module handles formatting of context bundles into prompt strings.
These functions belong in the context layer because they format agent-visible data.
"""
from __future__ import annotations

from src.context.schema import ContextBundle


def format_bounds_section(bundle: ContextBundle) -> str:
    """Format the parameter bounds section for prompts."""
    if not bundle.bounds:
        return ""
    lines = ["### Parameter Bounds\n"]
    for param, (low, high) in bundle.bounds.items():
        lines.append(f"  {param}: [{low}, {high}]\n")
    lines.append("\n")
    return "".join(lines)


def format_context_sections(bundle: ContextBundle) -> str:
    """Format all optional context sections (task, metric, bounds)."""
    sections = []
    if bundle.task_description:
        sections.append(f"### Task Description\n{bundle.task_description}\n\n")
    if bundle.metric_description:
        sections.append(f"### Evaluation Metric\n{bundle.metric_description}\n\n")
    sections.append(format_bounds_section(bundle))
    return "".join(sections)
