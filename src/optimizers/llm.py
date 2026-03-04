"""LLM optimizer - wraps existing direct LLM proposal logic."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.optimizers.base import BaseOptimizer, OptimizerConfig

if TYPE_CHECKING:
    from src.benchmarks.base import BaseBenchmark, IterationResult


class LLMOptimizer(BaseOptimizer):
    """LLM-based optimizer that uses language models to propose configurations.

    This optimizer wraps the existing _direct_llm_propose logic from
    BaseBenchmark, providing the same behavior through the optimizer interface.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        integer_keys: set,
        is_higher_better: bool,
        config: OptimizerConfig,
        benchmark: "BaseBenchmark",
    ):
        """Initialize LLM optimizer.

        Args:
            param_bounds: Dict mapping param names to (low, high) bounds
            integer_keys: Set of param names that should be integers
            is_higher_better: True if higher scores are better
            config: Optimizer configuration
            benchmark: Benchmark instance for prompt building and LLM calls
        """
        super().__init__(param_bounds, integer_keys, is_higher_better, config)
        self.benchmark = benchmark

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "llm"

    def propose(
        self,
        current_config: Dict[str, Any],
        last_score: float,
        history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
        """Propose configuration using LLM.

        This method converts the simplified history format back to
        IterationResult format and delegates to the benchmark's
        _direct_llm_propose method.

        Args:
            current_config: Current hyperparameter configuration
            last_score: Score from the last evaluation
            history: List of dicts with 'config' and 'score' keys

        Returns:
            Tuple of (proposal, source_label, token_usage_or_none)
        """
        # Convert simplified history to IterationResult format
        from src.benchmarks.base import IterationResult

        iteration_history: List[IterationResult] = []
        for i, entry in enumerate(history):
            # Create a minimal IterationResult with the required fields
            result = IterationResult(
                step=i,
                config=entry["config"],
                metrics={self._get_primary_metric_key(): entry["score"]},
                proposal_source="history",
            )
            iteration_history.append(result)

        # Build last_metrics dict from last_score
        last_metrics = {self._get_primary_metric_key(): last_score}

        # Call the benchmark's _direct_llm_propose method
        proposal, usage, parse_failure, clamp_events = self.benchmark._direct_llm_propose(
            current_config, last_metrics, iteration_history
        )

        if proposal:
            return proposal, "llm", usage
        return {}, "heuristic", usage

    def _get_primary_metric_key(self) -> str:
        """Get the primary metric key for this benchmark.

        We need a metric key to reconstruct metrics dicts from scores.
        The actual key doesn't matter since _get_primary_score extracts it.
        """
        return "score"
