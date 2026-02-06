"""Trace loading utilities for experiment analysis.

This module provides functions to parse JSONL trace files into structured
data suitable for analysis in pandas DataFrames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.constants import BENCHMARK_METADATA


def load_run(path: Path) -> dict:
    """Load single JSONL file, return parsed run dict.

    Args:
        path: Path to the JSONL trace file.

    Returns:
        Dictionary containing:
        - run_id: str
        - benchmark: str
        - seed: int
        - experiment_tags: dict (history_window, show_task, etc.)
        - events: list of all events
        - final_score: float (from run.end)
        - best_score: float (min/max of history depending on direction)
        - best_step: int
        - total_tokens: int
        - total_api_cost: float
        - total_latency_sec: float
        - convergence_curve: list of scores per step
    """
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        raise ValueError(f"Empty trace file: {path}")

    # Parse run.start event
    run_start = next((e for e in events if e["event_type"] == "run.start"), None)
    if run_start is None:
        raise ValueError(f"No run.start event found in {path}")

    run_id = run_start["run_id"]
    benchmark = run_start["task_id"]
    experiment_tags = parse_experiment_tags(run_start)
    seed = run_start["details"].get("seed", 0)

    # Parse run.end event
    run_end = next((e for e in events if e["event_type"] == "run.end"), None)
    if run_end is None:
        raise ValueError(f"No run.end event found in {path}")

    details = run_end["details"]
    final_score = details.get("final_metric")
    best_step_idx = details.get("best_step_idx", 0)
    total_tokens = details.get("total_tokens", 0)
    total_api_cost = details.get("total_api_cost", 0.0)
    total_latency_sec = details.get("total_latency_sec", 0.0)

    # Extract convergence curve from history in run.end
    history = details.get("extra", {}).get("history", [])
    metric_key = BENCHMARK_METADATA.get(benchmark, {}).get("metric", "mae")
    convergence_curve = []
    for step_data in history:
        metrics = step_data.get("metrics", {})
        score = metrics.get(metric_key)
        if score is not None:
            convergence_curve.append(score)

    # Compute best score
    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")
    if convergence_curve:
        if direction == "min":
            best_score = min(convergence_curve)
        else:
            best_score = max(convergence_curve)
    else:
        best_score = final_score

    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "seed": seed,
        "experiment_tags": experiment_tags,
        "events": events,
        "final_score": final_score,
        "best_score": best_score,
        "best_step": best_step_idx,
        "total_tokens": total_tokens,
        "total_api_cost": total_api_cost,
        "total_latency_sec": total_latency_sec,
        "convergence_curve": convergence_curve,
    }


def parse_experiment_tags(run_start_event: dict) -> dict:
    """Extract experiment tags from run.start event.

    Args:
        run_start_event: The run.start event dictionary.

    Returns:
        Dictionary of experiment tags (history_window, show_task,
        show_metric, show_resources, model, temperature).
    """
    details = run_start_event.get("details", {})
    tags = details.get("experiment_tags", {})
    return {
        "history_window": tags.get("history_window", 0),
        "show_task": tags.get("show_task", False),
        "show_metric": tags.get("show_metric", False),
        "show_resources": tags.get("show_resources", False),
        "model": tags.get("model", "unknown"),
        "temperature": tags.get("temperature", 0),
    }


def _make_config_id(tags: dict) -> str:
    """Generate config_id from experiment tags.

    Format: hw{history_window}_t{show_task}_m{show_metric}_r{show_resources}
    Boolean values are encoded as 1/0.

    Args:
        tags: Experiment tags dictionary.

    Returns:
        Config ID string (e.g., "hw5_t1_m0_r0").
    """
    hw = tags.get("history_window", 0)
    t = 1 if tags.get("show_task", False) else 0
    m = 1 if tags.get("show_metric", False) else 0
    r = 1 if tags.get("show_resources", False) else 0
    return f"hw{hw}_t{t}_m{m}_r{r}"


def load_all_runs(experiment_dir: str | Path, benchmark: str) -> pd.DataFrame:
    """Load all runs from a timestamped experiment directory.

    Args:
        experiment_dir: Path to experiment dir
            (e.g., traces/nomad_context_axes_v1__2026-02-06T00-04-37Z)
        benchmark: Benchmark name for metadata lookup.

    Returns:
        DataFrame with columns:
        - run_id: str
        - benchmark: str
        - seed: int
        - config_id: str (e.g., "hw5_t1_m0_r0")
        - history_window: int
        - show_task: bool
        - show_metric: bool
        - show_resources: bool
        - final_score: float
        - best_score: float
        - best_step: int
        - total_tokens: int
        - total_api_cost: float
        - total_latency_sec: float
        - convergence_curve: list of scores per step
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    jsonl_files = sorted(experiment_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {experiment_dir}")

    rows = []
    for jsonl_path in jsonl_files:
        try:
            run_data = load_run(jsonl_path)
            tags = run_data["experiment_tags"]
            rows.append({
                "run_id": run_data["run_id"],
                "benchmark": run_data["benchmark"],
                "seed": run_data["seed"],
                "config_id": _make_config_id(tags),
                "history_window": tags["history_window"],
                "show_task": tags["show_task"],
                "show_metric": tags["show_metric"],
                "show_resources": tags["show_resources"],
                "final_score": run_data["final_score"],
                "best_score": run_data["best_score"],
                "best_step": run_data["best_step"],
                "total_tokens": run_data["total_tokens"],
                "total_api_cost": run_data["total_api_cost"],
                "total_latency_sec": run_data["total_latency_sec"],
                "convergence_curve": run_data["convergence_curve"],
            })
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load {jsonl_path}: {e}")
            continue

    if not rows:
        raise ValueError(f"No valid runs loaded from {experiment_dir}")

    return pd.DataFrame(rows)
