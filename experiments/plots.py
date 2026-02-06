"""Visualization utilities for experiment results.

This module provides functions to generate plots for analyzing
experiment configurations and their performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.constants import BENCHMARK_METADATA


def plot_convergence_curves(
    df: pd.DataFrame,
    output_path: Path,
    benchmark: str = "nomad",
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot score vs step for each config (with seed variance bands).

    Args:
        df: DataFrame from load_all_runs with convergence_curve column.
        output_path: Path to save the figure.
        benchmark: Benchmark name for axis labels.
        figsize: Figure size tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric = BENCHMARK_METADATA.get(benchmark, {}).get("metric", "score")
    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique configs
    configs = df["config_id"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs)))

    for config_id, color in zip(configs, colors):
        config_df = df[df["config_id"] == config_id]
        curves = config_df["convergence_curve"].tolist()

        if not curves or all(len(c) == 0 for c in curves):
            continue

        # Find max length
        max_len = max(len(c) for c in curves)

        # Pad curves to same length
        padded_curves = []
        for curve in curves:
            if len(curve) < max_len:
                # Pad with last value
                padded = curve + [curve[-1]] * (max_len - len(curve))
            else:
                padded = curve[:max_len]
            padded_curves.append(padded)

        curves_array = np.array(padded_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        steps = np.arange(len(mean_curve))

        # Plot mean line
        ax.plot(steps, mean_curve, label=config_id, color=color, linewidth=1.5)

        # Plot variance band
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Convergence Curves by Configuration ({benchmark})", fontsize=14)

    # Place legend outside
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    # Invert y-axis if lower is better
    if direction == "min":
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_vs_cost(
    df: pd.DataFrame,
    output_path: Path,
    benchmark: str = "nomad",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Scatter: final_score vs total_api_cost, colored by config.

    Args:
        df: DataFrame from load_all_runs or aggregate_by_config.
        output_path: Path to save the figure.
        benchmark: Benchmark name for axis labels.
        figsize: Figure size tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric = BENCHMARK_METADATA.get(benchmark, {}).get("metric", "score")
    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

    fig, ax = plt.subplots(figsize=figsize)

    # Determine column names based on DataFrame structure
    if "final_score_mean" in df.columns:
        score_col = "final_score_mean"
        cost_col = "total_api_cost_mean"
    else:
        score_col = "final_score"
        cost_col = "total_api_cost"

    # Create scatter plot
    configs = df["config_id"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs)))
    config_colors = dict(zip(configs, colors))

    for config_id in configs:
        config_df = df[df["config_id"] == config_id]
        ax.scatter(
            config_df[cost_col] * 1000,  # Convert to millicents for readability
            config_df[score_col],
            label=config_id,
            color=config_colors[config_id],
            alpha=0.7,
            s=100,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("API Cost (millicents)", fontsize=12)
    ax.set_ylabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Score vs API Cost ({benchmark})", fontsize=14)

    # Place legend outside
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    # Invert y-axis if lower is better
    if direction == "min":
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_config_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    benchmark: str = "nomad",
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Heatmap: axes combinations vs metric value.

    Creates a 2D heatmap showing the metric value for different
    combinations of context axes.

    Args:
        df: DataFrame from load_all_runs or aggregate_by_config.
        metric: Metric column to visualize.
        output_path: Path to save the figure.
        benchmark: Benchmark name for labels.
        figsize: Figure size tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")

    # Determine metric column
    if f"{metric}_mean" in df.columns:
        metric_col = f"{metric}_mean"
    elif metric in df.columns:
        metric_col = metric
    else:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")

    # Aggregate if not already aggregated
    if "final_score_mean" not in df.columns:
        agg_df = df.groupby(["history_window", "show_task", "show_metric", "show_resources"]).agg({
            metric_col: "mean"
        }).reset_index()
    else:
        agg_df = df.copy()

    # Create pivot table for heatmap
    # Combine show_task, show_metric, show_resources into a single label
    agg_df["context_flags"] = agg_df.apply(
        lambda r: f"T{int(r['show_task'])}_M{int(r['show_metric'])}_R{int(r['show_resources'])}",
        axis=1
    )

    pivot = agg_df.pivot_table(
        index="context_flags",
        columns="history_window",
        values=metric_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Choose colormap based on direction
    cmap = "RdYlGn_r" if direction == "min" else "RdYlGn"

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": metric.upper()},
    )

    ax.set_xlabel("History Window", fontsize=12)
    ax.set_ylabel("Context Flags (T=Task, M=Metric, R=Resources)", fontsize=12)
    ax.set_title(f"{metric.upper()} by Configuration ({benchmark})", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_axis_effects(
    df: pd.DataFrame,
    output_path: Path,
    benchmark: str = "nomad",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Bar chart: marginal effect of each axis on final score.

    Shows the impact of each context axis on the final score.

    Args:
        df: DataFrame from load_all_runs.
        output_path: Path to save the figure.
        benchmark: Benchmark name for labels.
        figsize: Figure size tuple.
    """
    from experiments.analysis import compute_axis_effects

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effects_df = compute_axis_effects(df, benchmark)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    axes = effects_df["axis"].tolist()
    effects = effects_df["effect"].tolist()

    # Color bars based on positive/negative effect
    colors = ["green" if e > 0 else "red" for e in effects]

    bars = ax.bar(axes, effects, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax.annotate(
            f"{effect:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Context Axis", fontsize=12)
    ax.set_ylabel("Effect on Score (positive = better)", fontsize=12)
    ax.set_title(f"Marginal Effect of Each Axis ({benchmark})", fontsize=14)

    # Format x-axis labels
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([a.replace("_", " ").title() for a in axes], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(
    df: pd.DataFrame,
    output_path: Path,
    benchmark: str = "nomad",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot score vs cost with Pareto frontier highlighted.

    Args:
        df: Aggregated DataFrame with mean scores and costs.
        output_path: Path to save the figure.
        benchmark: Benchmark name for axis labels.
        figsize: Figure size tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direction = BENCHMARK_METADATA.get(benchmark, {}).get("direction", "min")
    metric = BENCHMARK_METADATA.get(benchmark, {}).get("metric", "score")

    # Determine column names
    if "final_score_mean" in df.columns:
        score_col = "final_score_mean"
        cost_col = "total_api_cost_mean"
    else:
        score_col = "final_score"
        cost_col = "total_api_cost"

    fig, ax = plt.subplots(figsize=figsize)

    # Compute Pareto frontier
    scores = df[score_col].values
    costs = df[cost_col].values
    config_ids = df["config_id"].values

    pareto_mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        is_pareto = True
        for j in range(len(df)):
            if i == j:
                continue
            # Check if j dominates i
            if direction == "min":
                # Lower score and lower cost is better
                if scores[j] <= scores[i] and costs[j] <= costs[i]:
                    if scores[j] < scores[i] or costs[j] < costs[i]:
                        is_pareto = False
                        break
            else:
                # Higher score and lower cost is better
                if scores[j] >= scores[i] and costs[j] <= costs[i]:
                    if scores[j] > scores[i] or costs[j] < costs[i]:
                        is_pareto = False
                        break
        pareto_mask[i] = is_pareto

    # Plot all points
    ax.scatter(
        costs * 1000,
        scores,
        c="lightgray",
        alpha=0.5,
        s=100,
        label="Non-Pareto",
    )

    # Plot Pareto points
    ax.scatter(
        costs[pareto_mask] * 1000,
        scores[pareto_mask],
        c="red",
        s=150,
        marker="*",
        label="Pareto Optimal",
        edgecolors="black",
        linewidth=0.5,
    )

    # Annotate Pareto points
    for i, (cost, score, config_id) in enumerate(zip(costs, scores, config_ids)):
        if pareto_mask[i]:
            ax.annotate(
                config_id,
                xy=(cost * 1000, score),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("API Cost (millicents)", fontsize=12)
    ax.set_ylabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Score vs Cost Pareto Frontier ({benchmark})", fontsize=14)
    ax.legend()

    if direction == "min":
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
