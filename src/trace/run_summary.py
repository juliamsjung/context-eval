"""Run-level summary schema for experimental analysis."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class RunSummary:
    """Structured summary of a single benchmark run."""
    # Identification
    benchmark: str
    seed: int
    run_id: str
    experiment_id: str  # Filesystem-safe identifier for grouping runs
    timestamp: str  # ISO-8601 format
    git_commit: Optional[str]  # Short SHA for reproducibility

    # Model configuration
    model_name: str
    temperature: float

    # Axes (experimental conditions)
    axis_signature: str  # e.g., "fd5_t1_m1_b1"
    feedback_depth: int
    show_task: bool
    show_metric: bool
    show_bounds: bool

    # Performance
    final_score: float
    best_score: float
    num_steps: int

    # Efficiency
    total_tokens: int
    total_cost: float

    # Diagnostics counts (aggregated)
    num_clamp_events: int
    num_parse_failures: int
    num_fallbacks: int
    num_truncations: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
