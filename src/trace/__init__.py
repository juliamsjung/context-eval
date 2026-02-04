"""Trace layer package for observability and debugging.

TRACE ONLY: This package handles all observability concerns including
structured logging, trace events, and run metadata. Data from this layer
must never flow into agent-visible context bundles.

Exports:
    - RunLogger: Run-scoped logger for JSONL events
    - start_run: Factory to create RunLogger instances
    - TRACE_ONLY_FIELDS: Fields that must never appear in context
    - TRACES_ROOT: Root directory for trace files
"""
from __future__ import annotations

from src.trace.logger import RunLogger, start_run, TRACES_ROOT
from src.trace.schema import TRACE_ONLY_FIELDS

__all__ = [
    "RunLogger",
    "start_run",
    "TRACES_ROOT",
    "TRACE_ONLY_FIELDS",
]
