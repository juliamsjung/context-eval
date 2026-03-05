"""Random search optimizer - uniform sampling baseline."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from src.optimizers.base import BaseOptimizer, OptimizerConfig


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer that samples uniformly from parameter bounds.

    This optimizer serves as a lower-bound baseline for comparing LLM
    optimization behavior. It samples each parameter uniformly from its
    bounds without using any history information.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        integer_keys: Set[str],
        is_higher_better: bool,
        config: OptimizerConfig,
    ):
        """Initialize random search optimizer.

        Args:
            param_bounds: Dict mapping param names to (low, high) bounds
            integer_keys: Set of param names that should be integers
            is_higher_better: True if higher scores are better
            config: Optimizer configuration with seed
        """
        super().__init__(param_bounds, integer_keys, is_higher_better, config)
        self._rng = random.Random(config.seed)

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "random"

    def propose(
        self,
        current_config: Dict[str, Any],
        last_score: float,
        history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
        """Propose a random configuration by uniform sampling.

        Args:
            current_config: Ignored - random search doesn't use current state
            last_score: Ignored - random search doesn't use scores
            history: Ignored - random search doesn't use history

        Returns:
            Tuple of (proposal, "random", None)
        """
        proposal = {}
        for param, (low, high) in self.param_bounds.items():
            value = self._rng.uniform(low, high)
            if param in self.integer_keys:
                value = int(round(value))
            proposal[param] = value
        return proposal, "random", None

    def reset(self) -> None:
        """Reset RNG to initial seed for reproducibility."""
        self._rng = random.Random(self.config.seed)
