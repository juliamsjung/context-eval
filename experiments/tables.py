"""Table generation utilities for experiment results.

This module provides functions to generate summary tables
and export them to CSV and LaTeX formats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summary_table(df: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Generate summary table: config -> score +/- std, cost, rank.

    Args:
        df: Aggregated DataFrame (from aggregate_by_config) with mean/std columns.
        benchmark: Optional benchmark name (for display purposes).

    Returns:
        DataFrame formatted for display with columns:
        - Config: config_id
        - Score: "mean +/- std" formatted string
        - Cost ($): formatted API cost
        - Tokens: total tokens
        - Rank: integer rank
    """
    result_rows = []

    for _, row in df.iterrows():
        # Format score as "mean +/- std"
        score_mean = row.get("final_score_mean", row.get("final_score", 0))
        score_std = row.get("final_score_std", 0)
        score_str = f"{score_mean:.4f} +/- {score_std:.4f}"

        # Format cost
        cost = row.get("total_api_cost_mean", row.get("total_api_cost", 0))
        cost_str = f"${cost:.4f}"

        # Format tokens
        tokens = row.get("total_tokens_mean", row.get("total_tokens", 0))
        tokens_str = f"{int(tokens):,}"

        # Get rank
        rank = row.get("rank", "N/A")

        result_rows.append({
            "Config": row["config_id"],
            "HW": row.get("history_window", "N/A"),
            "Task": "Y" if row.get("show_task", False) else "N",
            "Metric": "Y" if row.get("show_metric", False) else "N",
            "Resources": "Y" if row.get("show_resources", False) else "N",
            "Score": score_str,
            "Cost": cost_str,
            "Tokens": tokens_str,
            "Rank": rank,
        })

    return pd.DataFrame(result_rows)


def to_latex(df: pd.DataFrame, caption: str, label: str = "tab:results") -> str:
    """Export DataFrame to LaTeX table.

    Args:
        df: DataFrame to export.
        caption: Table caption.
        label: LaTeX label for referencing.

    Returns:
        LaTeX table string.
    """
    # Convert to LaTeX using pandas
    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(df.columns) - 1),
        escape=True,
    )

    # Add booktabs formatting
    latex = latex.replace("\\toprule", "\\toprule")
    latex = latex.replace("\\midrule", "\\midrule")
    latex = latex.replace("\\bottomrule", "\\bottomrule")

    return latex


def to_csv(df: pd.DataFrame, path: Path) -> None:
    """Export DataFrame to CSV.

    Args:
        df: DataFrame to export.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def axis_effects_table(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Format axis effects DataFrame for display.

    Args:
        effects_df: DataFrame from compute_axis_effects.

    Returns:
        Formatted DataFrame with columns:
        - Axis: axis name
        - Effect: formatted effect value
        - On Mean: mean score when axis is on
        - Off Mean: mean score when axis is off
    """
    result_rows = []

    for _, row in effects_df.iterrows():
        effect = row["effect"]
        effect_str = f"{effect:+.4f}"  # Include sign

        result_rows.append({
            "Axis": row["axis"].replace("_", " ").title(),
            "Effect": effect_str,
            "On Mean": f"{row['on_mean']:.4f}",
            "Off Mean": f"{row['off_mean']:.4f}",
        })

    return pd.DataFrame(result_rows)


def convergence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate convergence statistics table.

    Args:
        df: DataFrame with convergence stats (from compute_convergence_stats).

    Returns:
        DataFrame with convergence statistics per config.
    """
    # Aggregate convergence stats by config
    grouped = df.groupby("config_id").agg({
        "steps_to_90pct": ["mean", "std"],
        "improvement_rate": ["mean", "std"],
    })

    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    result_rows = []
    for _, row in grouped.iterrows():
        steps_mean = row.get("steps_to_90pct_mean", 0)
        steps_std = row.get("steps_to_90pct_std", 0)
        rate_mean = row.get("improvement_rate_mean", 0)
        rate_std = row.get("improvement_rate_std", 0)

        result_rows.append({
            "Config": row["config_id"],
            "Steps to 90%": f"{steps_mean:.1f} +/- {steps_std:.1f}" if pd.notna(steps_mean) else "N/A",
            "Improvement Rate": f"{rate_mean:.6f} +/- {rate_std:.6f}" if pd.notna(rate_mean) else "N/A",
        })

    return pd.DataFrame(result_rows)
