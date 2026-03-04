"""Optimizer strategies for hyperparameter search."""
from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING

from src.optimizers.base import BaseOptimizer, OptimizerConfig
from src.optimizers.random import RandomSearchOptimizer

if TYPE_CHECKING:
    from src.benchmarks.base import BaseBenchmark


def create_optimizer(
    optimizer_type: str,
    param_bounds: Dict[str, Tuple[float, float]],
    integer_keys: set,
    is_higher_better: bool,
    config: OptimizerConfig,
    benchmark: "BaseBenchmark" = None,
) -> BaseOptimizer:
    """Factory function to create an optimizer.

    Args:
        optimizer_type: Type of optimizer ('llm', 'random')
        param_bounds: Dict mapping param names to (low, high) bounds
        integer_keys: Set of param names that should be integers
        is_higher_better: True if higher scores are better
        config: Optimizer configuration
        benchmark: Benchmark instance (required for LLM optimizer)

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    if optimizer_type == "random":
        return RandomSearchOptimizer(
            param_bounds=param_bounds,
            integer_keys=integer_keys,
            is_higher_better=is_higher_better,
            config=config,
        )
    elif optimizer_type == "llm":
        from src.optimizers.llm import LLMOptimizer
        if benchmark is None:
            raise ValueError("benchmark is required for LLM optimizer")
        return LLMOptimizer(
            param_bounds=param_bounds,
            integer_keys=integer_keys,
            is_higher_better=is_higher_better,
            config=config,
            benchmark=benchmark,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


__all__ = [
    "BaseOptimizer",
    "OptimizerConfig",
    "RandomSearchOptimizer",
    "create_optimizer",
]
