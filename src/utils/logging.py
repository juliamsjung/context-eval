"""Run logging utilities for JSONL trace output.

BACKWARDS COMPATIBILITY SHIM: This module re-exports from src.trace
to maintain backwards compatibility with existing imports. New code
should import directly from src.trace.

Example (legacy):
    from src.utils.logging import RunLogger, start_run

Example (preferred):
    from src.trace import RunLogger, start_run
"""
from __future__ import annotations

# Re-export everything from the new trace package location
from src.trace.logger import (
    RunLogger,
    start_run,
    TRACES_ROOT,
    _now_iso,
    _default_run_id,
)

__all__ = [
    "RunLogger",
    "start_run",
    "TRACES_ROOT",
    "_now_iso",
    "_default_run_id",
]
