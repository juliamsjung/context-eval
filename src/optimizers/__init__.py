"""Optimizer strategies for hyperparameter search."""
from __future__ import annotations

from typing import Dict, Set, Tuple

from src.optimizers.base import BaseOptimizer, OptimizerConfig
from src.optimizers.random import RandomSearchOptimizer


def create_optimizer(
    optimizer_type: str,
    param_bounds: Dict[str, Tuple[float, float]],
    integer_keys: Set[str],
    is_higher_better: bool,
    config: OptimizerConfig,
) -> BaseOptimizer:
    """Factory function to create an optimizer.

    Args:
        optimizer_type: Type of optimizer ('random'). Note: 'llm' uses direct
            path in BaseBenchmark and does not go through this factory.
        param_bounds: Dict mapping param names to (low, high) bounds
        integer_keys: Set of param names that should be integers
        is_higher_better: True if higher scores are better
        config: Optimizer configuration

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer_type is not recognized or not supported
    """
    if optimizer_type == "random":
        return RandomSearchOptimizer(
            param_bounds=param_bounds,
            integer_keys=integer_keys,
            is_higher_better=is_higher_better,
            config=config,
        )
    elif optimizer_type == "llm":
        raise ValueError(
            "LLM optimizer uses direct path in BaseBenchmark.propose_config(), "
            "not the optimizer abstraction"
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


__all__ = [
    "BaseOptimizer",
    "OptimizerConfig",
    "RandomSearchOptimizer",
    "create_optimizer",
]
