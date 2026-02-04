"""Trace-only field definitions for observability layer.

This module defines constants that identify fields which belong exclusively
to the trace/observability layer and must never appear in agent-visible context.
"""
from __future__ import annotations

from typing import FrozenSet

# TRACE ONLY: Fields that are logged for analysis and debugging but must never
# appear in agent-visible context bundles. These include:
# - token_usage: LLM API usage metrics
# - api_cost: Computed cost from token usage
# - latency_sec: API call timing
# - total_api_cost, total_tokens, total_latency_sec: Aggregate run metrics
# - experiment_tags: Run-level metadata for analysis grouping
# - config_hash: Internal tracking identifier
# - input_tokens, output_tokens: Detailed token breakdowns
#
# Note: run_id and timestamp are excluded from this set because they are
# structural envelope fields that never belong in context bundles and would
# be caught by structural validation if somehow present.
TRACE_ONLY_FIELDS: FrozenSet[str] = frozenset({
    "token_usage",
    "api_cost",
    "latency_sec",
    "total_api_cost",
    "total_tokens",
    "total_latency_sec",
    "experiment_tags",
    "config_hash",
    "input_tokens",
    "output_tokens",
})
