"""Abstract base class for optimizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    seed: int = 0


class BaseOptimizer(ABC):
    """Abstract base class for optimization strategies.

    Optimizers propose hyperparameter configurations based on
    history of previous evaluations.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        integer_keys: set,
        is_higher_better: bool,
        config: OptimizerConfig,
    ):
        """Initialize the optimizer.

        Args:
            param_bounds: Dict mapping param names to (low, high) bounds
            integer_keys: Set of param names that should be integers
            is_higher_better: True if higher scores are better
            config: Optimizer configuration
        """
        self.param_bounds = param_bounds
        self.integer_keys = integer_keys
        self.is_higher_better = is_higher_better
        self.config = config

    @abstractmethod
    def propose(
        self,
        current_config: Dict[str, Any],
        last_score: float,
        history: List[Dict[str, Any]],  # [{config, score}, ...]
    ) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
        """Propose a new configuration.

        Args:
            current_config: Current hyperparameter configuration
            last_score: Score from the last evaluation
            history: List of previous evaluations with config and score

        Returns:
            Tuple of (proposal, source_label, token_usage_or_none)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer name (e.g., 'llm', 'random')."""
        ...

    def reset(self) -> None:
        """Reset optimizer state. Called at the start of a new run."""
        pass
