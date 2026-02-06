"""Core analysis functions for experiment results.

This module provides functions to aggregate, compute statistics,
and rank experiment configurations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiments.constants import BENCHMARK_METADATA


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Group by config_id (aggregates across seeds).

    Computes mean/std across seeds for:
    - final_score, best_score
    - total_tokens, total_api_cost

    Args:
        df: DataFrame from load_all_runs with individual run data.

    Returns:
        DataFrame with columns:
        - config_id: str
        - history_window: int
        - show_task: bool
        - show_metric: bool
        - show_resources: bool
        - n_seeds: int (number of seeds)
        - final_score_mean: float
        - final_score_std: float
        - best_score_mean: float
        - best_score_std: float
        - total_tokens_mean: float
        - total_tokens_std: float
        - total_api_cost_mean: float
        - total_api_cost_std: float
    """
    # Group by config_id
    grouped = df.groupby("config_id")

    # Aggregate numeric columns
    agg_dict = {
        "final_score": ["mean", "std", "count"],
        "best_score": ["mean", "std"],
        "total_tokens": ["mean", "std"],
        "total_api_cost": ["mean", "std"],
        "total_latency_sec": ["mean", "std"],
        "history_window": "first",
        "show_task": "first",
        "show_metric": "first",
        "show_resources": "first",
    }

    result = grouped.agg(agg_dict)

    # Flatten column names
    result.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in result.columns.values
    ]

    # Rename columns
    result = result.rename(columns={
        "final_score_count": "n_seeds",
        "history_window_first": "history_window",
        "show_task_first": "show_task",
        "show_metric_first": "show_metric",
        "show_resources_first": "show_resources",
    })

    # Reset index to make config_id a column
    result = result.reset_index()

    # Fill NaN std values with 0 (happens when n=1)
    std_cols = [c for c in result.columns if c.endswith("_std")]
    result[std_cols] = result[std_cols].fillna(0)

    return result


def compute_convergence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute convergence metrics per config.

    Computes:
    - steps_to_threshold: Steps to reach 90% of best score
    - improvement_rate: Score improvement per step

    Args:
        df: DataFrame from load_all_runs with convergence_curve column.

    Returns:
        DataFrame with additional columns:
        - steps_to_90pct: int (steps to reach 90% of best)
        - improvement_rate: float (score delta per step)
    """
    results = []

    for _, row in df.iterrows():
        curve = row.get("convergence_curve", [])
        config_id = row["config_id"]
        seed = row["seed"]

        if not curve or len(curve) < 2:
            results.append({
                "config_id": config_id,
                "seed": seed,
                "steps_to_90pct": None,
                "improvement_rate": None,
            })
            continue

        best_score = row["best_score"]
        initial_score = curve[0]

        # Calculate 90% threshold (depends on direction)
        # For minimization: threshold = initial - 0.9 * (initial - best)
        # For maximization: threshold = initial + 0.9 * (best - initial)
        benchmark = row.get("benchmark", "nomad")
        direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

        if direction == "min":
            threshold = initial_score - 0.9 * (initial_score - best_score)
            steps_to_90pct = next(
                (i for i, s in enumerate(curve) if s <= threshold),
                len(curve) - 1
            )
        else:
            threshold = initial_score + 0.9 * (best_score - initial_score)
            steps_to_90pct = next(
                (i for i, s in enumerate(curve) if s >= threshold),
                len(curve) - 1
            )

        # Improvement rate: total improvement / number of steps
        total_improvement = abs(curve[-1] - curve[0])
        improvement_rate = total_improvement / (len(curve) - 1) if len(curve) > 1 else 0

        results.append({
            "config_id": config_id,
            "seed": seed,
            "steps_to_90pct": steps_to_90pct,
            "improvement_rate": improvement_rate,
        })

    conv_df = pd.DataFrame(results)

    # Merge back with original df
    result = df.merge(conv_df, on=["config_id", "seed"], how="left")

    return result


def rank_configs(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Rank configurations by primary metric using BENCHMARK_METADATA direction.

    Args:
        df: Aggregated DataFrame (from aggregate_by_config) or raw DataFrame.
        benchmark: Benchmark name for direction lookup.

    Returns:
        DataFrame with additional 'rank' column (1 = best).
    """
    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

    # Determine score column
    if "final_score_mean" in df.columns:
        score_col = "final_score_mean"
    elif "final_score" in df.columns:
        score_col = "final_score"
    else:
        raise ValueError("No score column found in DataFrame")

    # Rank: ascending=True for min (lower is better), False for max
    ascending = direction == "min"
    result = df.copy()
    result["rank"] = result[score_col].rank(ascending=ascending, method="min").astype(int)

    # Sort by rank
    result = result.sort_values("rank")

    return result


def compute_axis_effects(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Compute marginal effects of each axis on the final score.

    For each axis (history_window, show_task, show_metric, show_resources),
    computes the average score difference when that axis is "on" vs "off".

    Args:
        df: DataFrame from load_all_runs.
        benchmark: Benchmark name for direction lookup.

    Returns:
        DataFrame with columns:
        - axis: str (name of the axis)
        - effect: float (positive = improves score, negative = hurts)
        - on_mean: float (mean score when axis is "on")
        - off_mean: float (mean score when axis is "off")
    """
    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

    results = []
    for axis in ["show_task", "show_metric", "show_resources"]:
        on_mask = df[axis] == True  # noqa: E712
        off_mask = df[axis] == False  # noqa: E712

        on_mean = df.loc[on_mask, "final_score"].mean()
        off_mean = df.loc[off_mask, "final_score"].mean()

        # Effect: positive means the axis improves score
        if direction == "min":
            effect = off_mean - on_mean  # Lower is better, so positive if on < off
        else:
            effect = on_mean - off_mean  # Higher is better, so positive if on > off

        results.append({
            "axis": axis,
            "effect": effect,
            "on_mean": on_mean,
            "off_mean": off_mean,
        })

    # Handle history_window (numeric axis)
    hw_values = df["history_window"].unique()
    if len(hw_values) > 1:
        # Compare history_window > 0 vs history_window == 0
        on_mask = df["history_window"] > 0
        off_mask = df["history_window"] == 0

        on_mean = df.loc[on_mask, "final_score"].mean()
        off_mean = df.loc[off_mask, "final_score"].mean()

        if direction == "min":
            effect = off_mean - on_mean
        else:
            effect = on_mean - off_mean

        results.append({
            "axis": "history_window",
            "effect": effect,
            "on_mean": on_mean,
            "off_mean": off_mean,
        })

    return pd.DataFrame(results)
