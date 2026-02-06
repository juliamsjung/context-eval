"""Experiments analysis module for analyzing trace results from experiment grids."""

from experiments.constants import BENCHMARK_METADATA
from experiments.loader import load_run, load_all_runs, parse_experiment_tags
from experiments.analysis import (
    aggregate_by_config,
    compute_convergence_stats,
    rank_configs,
)
from experiments.tables import summary_table, to_latex, to_csv
from experiments.plots import (
    plot_convergence_curves,
    plot_score_vs_cost,
    plot_config_heatmap,
    plot_axis_effects,
)

__all__ = [
    "BENCHMARK_METADATA",
    "load_run",
    "load_all_runs",
    "parse_experiment_tags",
    "aggregate_by_config",
    "compute_convergence_stats",
    "rank_configs",
    "summary_table",
    "to_latex",
    "to_csv",
    "plot_convergence_curves",
    "plot_score_vs_cost",
    "plot_config_heatmap",
    "plot_axis_effects",
]
