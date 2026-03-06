"""Batch evaluation runner for landscape characterization.

Evaluates a set of configurations against a benchmark's training
pipeline and records the results for downstream analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm


class LandscapeRunner:
    """Batch-evaluate configurations using a benchmark's training pipeline.

    This class takes a training function and a score extractor, evaluates
    each configuration, and saves the results for stratified selection.

    Args:
        run_training: Callable that takes a config dict and returns metrics dict.
        score_extractor: Callable that extracts the primary score from metrics.
        higher_is_better: Whether higher scores indicate better performance.
        benchmark_name: Name of the benchmark for output file naming.
    """

    def __init__(
        self,
        run_training: Callable[[Dict[str, Any]], Dict[str, float]],
        score_extractor: Callable[[Dict[str, float]], float],
        higher_is_better: bool,
        benchmark_name: str,
    ) -> None:
        self.run_training = run_training
        self.score_extractor = score_extractor
        self.higher_is_better = higher_is_better
        self.benchmark_name = benchmark_name

    def evaluate(
        self,
        configs: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate all configurations and return scored results.

        Args:
            configs: List of config dicts to evaluate.
            output_path: Optional path to save results JSON. If None,
                         defaults to logs/landscape/{benchmark}_landscape.json.

        Returns:
            List of dicts with keys: config, metrics, primary_score.
        """
        results: List[Dict[str, Any]] = []

        for i, config in enumerate(tqdm(configs, desc=f"Evaluating {self.benchmark_name} landscape")):
            try:
                metrics = self.run_training(config)
                score = self.score_extractor(metrics)
                results.append({
                    "index": i,
                    "config": config,
                    "metrics": metrics,
                    "primary_score": score,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "config": config,
                    "metrics": {},
                    "primary_score": None,
                    "error": str(e),
                })

        # Save results
        if output_path is None:
            output_path = Path("logs/landscape") / f"{self.benchmark_name}_landscape.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "benchmark": self.benchmark_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(configs),
            "num_successful": sum(1 for r in results if r["error"] is None),
            "higher_is_better": self.higher_is_better,
            "results": results,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

        return results
