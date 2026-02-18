"""Context formatting utilities for prompt construction.

CONTEXT ONLY: This module handles formatting of context bundles into prompt strings.
These functions belong in the context layer because they format agent-visible data.
"""
from __future__ import annotations

from src.context.schema import ContextBundle


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
