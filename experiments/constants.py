"""Constants for experiment analysis.

This module defines benchmark metadata and other constants used across
the experiments package.
"""

from __future__ import annotations

BENCHMARK_METADATA = {
    "nomad": {"metric": "mae", "direction": "min"},
    "toy_tabular": {"metric": "accuracy", "direction": "max"},
}
