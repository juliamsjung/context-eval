"""Behavioral stability metrics for agent configuration search trajectories.

Evaluates the quality of an agent's search process — independent of benchmark
performance — by measuring how erratically or redundantly it navigates the
hyperparameter space.

Metrics
-------
Configuration Churn
    The average normalized distance between consecutive hyperparameter
    configurations across a run.  A high churn value indicates large,
    potentially unfocused jumps between steps; a low value suggests the agent
    is refining in small increments.

Instability Score
    The fraction of steps in the trajectory that revisit a previously-seen
    configuration (exact duplicates, using a canonical JSON hash).  This
    captures oscillation / cycling behaviour: a score of 0 means every step
    was a novel configuration, while a score of 1 means every step was a
    repeat.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Tuple


class StabilityMetric:
    """Evaluate the stability of an agent's configuration search trajectory.

    All public methods are stateless; instantiate once and call repeatedly.

    Methods
    -------
    calculate_config_distance(config_a, config_b)
        Pairwise normalized distance between two configuration dicts.
    detect_oscillation(trace_history)
        Instability score for a full trace.
    evaluate_trace(trace)
        Combined report: churn + instability over a completed run.
    """

    # ------------------------------------------------------------------ #
    # Pairwise distance                                                    #
    # ------------------------------------------------------------------ #

    def calculate_config_distance(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Return the normalized distance between two configuration dicts.

        Each hyperparameter contributes independently:

        * **Numerical** (``int`` / ``float``): fractional difference relative
          to the larger absolute value, clamped to ``[0, 1]``.
          ``|a - b| / (max(|a|, |b|) + ε)``
        * **Categorical / string**: binary penalty — 0 if equal, 1 if changed.
        * **Missing in one dict**: treated as a categorical change (score 1.0).

        Parameters
        ----------
        config_a, config_b:
            Configuration dicts to compare.  Keys may differ; keys present in
            only one dict count as a change.

        Returns
        -------
        distance_score : float
            Summed distance across all keys (≥ 0).
        changed_keys : list[str]
            Names of hyperparameters that differed.
        """
        distance_score = 0.0
        changed_keys: List[str] = []

        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)

            if val_a == val_b:
                continue

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                denom = max(abs(val_a), abs(val_b)) + 1e-9
                distance_score += abs(val_a - val_b) / denom
            else:
                # Categorical or type-mismatched: binary penalty
                distance_score += 1.0

            changed_keys.append(key)

        return distance_score, changed_keys

    # ------------------------------------------------------------------ #
    # Oscillation / cycle detection                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _config_hash(config: Dict[str, Any]) -> str:
        """Canonical string hash for a configuration dict.

        Uses ``json.dumps`` with sorted keys so that insertion-order
        differences in the dict do not produce false mismatches.
        """
        return json.dumps(config, sort_keys=True)

    def detect_oscillation(self, trace_history: List[Dict[str, Any]]) -> float:
        """Compute the Instability Score for a trajectory.

        The score is the fraction of steps that revisit a previously-seen
        configuration::

            instability = repeated_steps / total_steps

        A repeated step is any step whose configuration has appeared at least
        once before in the trace.  The first occurrence of a configuration is
        not penalised; only subsequent duplicates are counted.

        Parameters
        ----------
        trace_history:
            List of step dicts.  Each dict must contain a ``'config'`` key
            whose value is the hyperparameter dict for that step.

        Returns
        -------
        float
            Instability score in ``[0, 1]``.  Returns ``0.0`` for empty
            traces.
        """
        if not trace_history:
            return 0.0

        seen: set[str] = set()
        repeated_steps = 0

        for step in trace_history:
            config = step.get("config", step)  # graceful fallback
            h = self._config_hash(config) if isinstance(config, dict) else str(config)
            if h in seen:
                repeated_steps += 1
            else:
                seen.add(h)

        return repeated_steps / len(trace_history)

    # ------------------------------------------------------------------ #
    # Full-trace evaluation                                                #
    # ------------------------------------------------------------------ #

    def evaluate_trace(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a complete stability evaluation over a finished benchmark trace.

        Computes both the Instability Score (oscillation) and the average
        Configuration Churn (step-to-step distance).

        Parameters
        ----------
        trace:
            List of step dicts, each containing at minimum a ``'config'``
            key.  This matches the ``history`` list produced by
            ``BaseBenchmark.run()``.

        Returns
        -------
        dict with keys:

        ``instability_score`` : float
            Fraction of steps that revisit a prior configuration.
        ``average_churn`` : float
            Mean pairwise distance between consecutive configurations.
        ``total_churn`` : float
            Sum of all pairwise distances (unnormalised).
        ``total_steps`` : int
            Number of steps in the trace (including the baseline step 0).
        ``churn_steps`` : int
            Number of consecutive pairs evaluated (``total_steps - 1``).
        """
        instability_score = self.detect_oscillation(trace)

        total_churn = 0.0
        churn_steps = 0

        for i in range(len(trace) - 1):
            config_a = trace[i].get("config", {})
            config_b = trace[i + 1].get("config", {})
            dist, _ = self.calculate_config_distance(config_a, config_b)
            total_churn += dist
            churn_steps += 1

        avg_churn = total_churn / churn_steps if churn_steps > 0 else 0.0

        return {
            "instability_score": instability_score,
            "average_churn": avg_churn,
            "total_churn": total_churn,
            "total_steps": len(trace),
            "churn_steps": churn_steps,
        }
