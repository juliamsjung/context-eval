"""Stratified config selector for performance-based initialization.

Selects representative configurations at different performance levels
(P25, P50, P75) from landscape characterization results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StratifiedSelector:
    """Select P25/P50/P75 configs from landscape results.

    Sorts evaluated configs by performance and picks the median config
    from each performance stratum:
    - low (P25): median of bottom quartile
    - neutral (P50): overall median
    - high (P75): median of top quartile, excluding top 5% outliers

    Args:
        higher_is_better: Whether higher scores indicate better performance.
    """

    def __init__(self, higher_is_better: bool = True) -> None:
        self.higher_is_better = higher_is_better

    def select(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        benchmark_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Select stratified init configs from landscape results.

        Args:
            results: List of evaluated configs from LandscapeRunner.
            output_dir: Optional directory to save individual init config JSONs.
                        If None, defaults to logs/landscape/{benchmark}_init_configs/.
            benchmark_name: Benchmark name for default output directory.

        Returns:
            Dict with 'low', 'neutral', 'high' keys, each mapping to a dict
            with 'config', 'primary_score', and 'percentile' fields.
        """
        # Filter out failures
        valid = [r for r in results if r["error"] is None and r["primary_score"] is not None]

        if len(valid) < 4:
            raise ValueError(
                f"Need at least 4 successful evaluations for stratified selection, "
                f"got {len(valid)}."
            )

        # Sort by score (ascending = worst-to-best for higher_is_better,
        # descending = worst-to-best for lower_is_better)
        sorted_results = sorted(
            valid,
            key=lambda r: r["primary_score"],
            reverse=not self.higher_is_better,
        )

        n = len(sorted_results)

        # P25: median of bottom quartile (indices 0 to n//4)
        q1_end = n // 4
        low_idx = q1_end // 2

        # P50: overall median
        neutral_idx = n // 2

        # P75: median of top quartile, excluding top 5%
        # Top quartile starts at 3n/4, top 5% cutoff at 0.95*n
        q3_start = (3 * n) // 4
        top_cutoff = int(0.95 * n)
        # Median of range [q3_start, top_cutoff)
        high_idx = (q3_start + top_cutoff) // 2

        # Clamp indices
        low_idx = max(0, min(low_idx, n - 1))
        neutral_idx = max(0, min(neutral_idx, n - 1))
        high_idx = max(0, min(high_idx, n - 1))

        selections = {
            "low": {
                "config": sorted_results[low_idx]["config"],
                "primary_score": sorted_results[low_idx]["primary_score"],
                "percentile": round(low_idx / n * 100, 1),
            },
            "neutral": {
                "config": sorted_results[neutral_idx]["config"],
                "primary_score": sorted_results[neutral_idx]["primary_score"],
                "percentile": round(neutral_idx / n * 100, 1),
            },
            "high": {
                "config": sorted_results[high_idx]["config"],
                "primary_score": sorted_results[high_idx]["primary_score"],
                "percentile": round(high_idx / n * 100, 1),
            },
        }

        # Save individual config files for grid script consumption
        if output_dir is None and benchmark_name:
            output_dir = Path("logs/landscape") / f"{benchmark_name}_init_configs"

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for quality, data in selections.items():
                config_path = output_dir / f"{quality}.json"
                config_path.write_text(json.dumps(data["config"], indent=2))

            # Also save the full selection metadata
            meta_path = output_dir / "selection_metadata.json"
            meta_path.write_text(json.dumps(selections, indent=2))

        return selections
