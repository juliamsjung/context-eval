"""Sobol quasi-random sampler for landscape characterization.

Generates space-filling configurations across a benchmark's parameter
space using Sobol sequences, with support for log-scale and integer params.

Default pool size is 256 (power of 2) for exact Sobol sequence balance.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from scipy.stats.qmc import Sobol


class SobolSampler:
    """Generate quasi-random configurations via Sobol sequences.

    Sobol sequences provide better coverage of high-dimensional spaces
    than simple random sampling, ensuring no regions of high performance
    are missed during landscape characterization.

    Args:
        param_bounds: Dict mapping param names to (low, high) bounds.
        log_scale_params: Set of param names to sample in log-space.
        integer_keys: Set of param names that should be cast to int.
        seed: Random seed for the Sobol sequence.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        log_scale_params: Optional[Set[str]] = None,
        integer_keys: Optional[Set[str]] = None,
        seed: int = 0,
    ) -> None:
        self.param_bounds = param_bounds
        self.log_scale_params = log_scale_params or set()
        self.integer_keys = integer_keys or set()
        self.seed = seed

        # Stable key ordering for reproducibility
        self._param_names = sorted(param_bounds.keys())
        self._dimension = len(self._param_names)

        # Validate log-scale params have positive bounds
        for name in self.log_scale_params:
            lo, hi = self.param_bounds[name]
            if lo <= 0:
                raise ValueError(
                    f"Log-scale param '{name}' has non-positive lower bound {lo}. "
                    f"Log-scale requires bounds > 0."
                )

    def sample(self, n: int = 256) -> List[Dict[str, Any]]:
        """Generate n quasi-random configurations.

        Args:
            n: Number of samples to generate. Default is 256 (power of 2
               for exact Sobol balance). Will be rounded up to the next
               power of 2 internally if not already, but only the first
               n samples are returned.

        Returns:
            List of n config dictionaries with values within bounds.
        """
        # Sobol requires 2^m samples; generate enough and truncate
        m = max(1, math.ceil(math.log2(n)))
        n_sobol = 2**m

        # scramble=True is explicit to guard against future scipy default changes
        sampler = Sobol(d=self._dimension, seed=self.seed, scramble=True)
        # Generate unit hypercube samples in [0, 1]^d
        raw_samples = sampler.random(n_sobol)

        configs: List[Dict[str, Any]] = []
        for i in range(min(n, n_sobol)):
            config: Dict[str, Any] = {}
            for j, name in enumerate(self._param_names):
                lo, hi = self.param_bounds[name]
                unit_val = raw_samples[i, j]

                if name in self.log_scale_params:
                    # Map [0, 1] -> [log(lo), log(hi)] -> exp
                    log_lo = math.log(lo)
                    log_hi = math.log(hi)
                    val = math.exp(log_lo + unit_val * (log_hi - log_lo))
                else:
                    # Map [0, 1] -> [lo, hi]
                    val = lo + unit_val * (hi - lo)

                if name in self.integer_keys:
                    val = int(round(val))

                config[name] = val

            configs.append(config)

        return configs
