"""Stratified config selector for performance-based initialization.

Selects representative configurations from different performance strata
using normalized regret with guard bands for clean separation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StratifiedSelector:
    """Select configs from good/general/bad strata based on normalized regret.

    Defines three strata with guard bands to ensure clean separation:
    - good (high): r <= 0.20 (top 20% of pool) → label "high"
    - general (neutral): 0.45 <= r <= 0.55 (middle 10% of pool) → label "neutral"
    - bad (low): r >= 0.80 (bottom 20% of pool) → label "low"

    Normalized regret is defined as:
        r(x) = (loss(x) - loss_min) / (loss_max - loss_min)

    For higher-is-better metrics, loss = 1 - metric before computing regret.

    Args:
        higher_is_better: Whether higher scores indicate better performance.
    """

    # Stratum boundaries (normalized regret thresholds)
    GOOD_UPPER = 0.20      # r <= 0.20 for good stratum
    NEUTRAL_LOWER = 0.45   # 0.45 <= r <= 0.55 for neutral stratum
    NEUTRAL_UPPER = 0.55
    BAD_LOWER = 0.80       # r >= 0.80 for bad stratum

    def __init__(self, higher_is_better: bool = True) -> None:
        self.higher_is_better = higher_is_better

    def _compute_normalized_regret(
        self,
        score: float,
        score_min: float,
        score_max: float,
    ) -> float:
        """Compute normalized regret r in [0, 1].

        For higher-is-better: r = (score_max - score) / (score_max - score_min)
        For lower-is-better:  r = (score - score_min) / (score_max - score_min)

        Lower regret = better performance.
        """
        if score_max == score_min:
            return 0.5  # All scores identical

        if self.higher_is_better:
            # Higher score = lower regret
            return (score_max - score) / (score_max - score_min)
        else:
            # Lower score = lower regret
            return (score - score_min) / (score_max - score_min)

    def _select_median_from_stratum(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select the median config from a list of candidates (by regret)."""
        sorted_by_regret = sorted(candidates, key=lambda r: r["_regret"])
        median_idx = len(sorted_by_regret) // 2
        return sorted_by_regret[median_idx]

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
            with 'config', 'score', 'normalized_regret', 'percentile', 'stratum'.
        """
        # Filter out failures
        valid = [r for r in results if r["error"] is None and r["primary_score"] is not None]

        if len(valid) < 4:
            raise ValueError(
                f"Need at least 4 successful evaluations for stratified selection, "
                f"got {len(valid)}."
            )

        # Compute score range
        scores = [r["primary_score"] for r in valid]
        score_min = min(scores)
        score_max = max(scores)

        # Sort by score (worst to best) for percentile computation
        # Create shallow copies to avoid mutating input
        sorted_results = sorted(
            [dict(r) for r in valid],
            key=lambda r: r["primary_score"],
            reverse=not self.higher_is_better,
        )

        # Add normalized regret and percentile to copies
        n = len(sorted_results)
        for i, r in enumerate(sorted_results):
            r["_regret"] = self._compute_normalized_regret(
                r["primary_score"], score_min, score_max
            )
            r["_percentile"] = round((i / n) * 100, 1)

        # Partition into strata based on normalized regret
        good_stratum = [r for r in sorted_results if r["_regret"] <= self.GOOD_UPPER]
        neutral_stratum = [
            r for r in sorted_results
            if self.NEUTRAL_LOWER <= r["_regret"] <= self.NEUTRAL_UPPER
        ]
        bad_stratum = [r for r in sorted_results if r["_regret"] >= self.BAD_LOWER]

        # Validate we have configs in each stratum
        if not good_stratum:
            raise ValueError(
                f"No configs found in good stratum (r <= {self.GOOD_UPPER}). "
                f"Pool may be too small or have insufficient score variance."
            )
        if not neutral_stratum:
            raise ValueError(
                f"No configs found in neutral stratum ({self.NEUTRAL_LOWER} <= r <= {self.NEUTRAL_UPPER}). "
                f"Pool may be too small or have insufficient score variance."
            )
        if not bad_stratum:
            raise ValueError(
                f"No configs found in bad stratum (r >= {self.BAD_LOWER}). "
                f"Pool may be too small or have insufficient score variance."
            )

        # Select median from each stratum
        high_result = self._select_median_from_stratum(good_stratum)
        neutral_result = self._select_median_from_stratum(neutral_stratum)
        low_result = self._select_median_from_stratum(bad_stratum)

        # Build output format
        def format_selection(result: Dict[str, Any], stratum: str) -> Dict[str, Any]:
            return {
                "config": result["config"],
                "score": result["primary_score"],
                "normalized_regret": round(result["_regret"], 4),
                "percentile": result["_percentile"],
                "stratum": stratum,
            }

        selections = {
            "low": format_selection(low_result, "low"),
            "neutral": format_selection(neutral_result, "neutral"),
            "high": format_selection(high_result, "high"),
        }

        # Save individual config files for grid script consumption
        if output_dir is None and benchmark_name:
            output_dir = Path("logs/landscape") / f"{benchmark_name}_init_configs"

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for quality, data in selections.items():
                config_path = output_dir / f"{quality}.json"
                # Write full selection data (includes config, score, regret, etc.)
                config_path.write_text(json.dumps(data, indent=2))

            # Also save the full selection metadata
            meta_path = output_dir / "selection_metadata.json"
            meta_path.write_text(json.dumps(selections, indent=2))

        return selections
