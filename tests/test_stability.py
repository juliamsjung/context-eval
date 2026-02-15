"""Tests for src/metrics/stability.py — StabilityMetric.

Covers:
- calculate_config_distance  (numerical, categorical, missing keys, identical)
- detect_oscillation         (no repeats, partial repeats, all repeats, empty)
- evaluate_trace             (output shape, values, edge cases)
- Integration: StabilityMetric is importable from src.metrics
"""
from __future__ import annotations

import pytest

from src.metrics import StabilityMetric
from src.metrics.stability import StabilityMetric as StabilityMetricDirect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(*configs):
    """Wrap a sequence of config dicts into the trace format expected by
    detect_oscillation / evaluate_trace."""
    return [{"config": c} for c in configs]


# ---------------------------------------------------------------------------
# TestImport
# ---------------------------------------------------------------------------

class TestImport:
    """StabilityMetric is reachable via the public package interface."""

    def test_importable_from_package(self):
        assert StabilityMetric is StabilityMetricDirect

    def test_instantiation(self):
        m = StabilityMetric()
        assert m is not None


# ---------------------------------------------------------------------------
# TestConfigDistance
# ---------------------------------------------------------------------------

class TestConfigDistance:
    """calculate_config_distance — pairwise hyperparameter distance."""

    def setup_method(self):
        self.m = StabilityMetric()

    # --- identical configs --------------------------------------------------

    def test_identical_configs_zero_distance(self):
        cfg = {"lr": 0.1, "depth": 5, "optimizer": "adam"}
        dist, changed = self.m.calculate_config_distance(cfg, cfg)
        assert dist == 0.0
        assert changed == []

    def test_empty_configs_zero_distance(self):
        dist, changed = self.m.calculate_config_distance({}, {})
        assert dist == 0.0
        assert changed == []

    # --- numerical params ---------------------------------------------------

    def test_numerical_distance_same_value(self):
        dist, changed = self.m.calculate_config_distance({"lr": 0.1}, {"lr": 0.1})
        assert dist == 0.0
        assert changed == []

    def test_numerical_distance_half_of_max(self):
        # |0.1 - 0.05| / max(0.1, 0.05) = 0.05 / 0.1 = 0.5
        dist, changed = self.m.calculate_config_distance({"lr": 0.1}, {"lr": 0.05})
        assert pytest.approx(dist, abs=1e-6) == 0.5
        assert changed == ["lr"]

    def test_numerical_distance_one_is_zero(self):
        # |1.0 - 0.0| / (max(1.0, 0.0) + eps) ≈ 1.0
        dist, changed = self.m.calculate_config_distance({"lr": 1.0}, {"lr": 0.0})
        assert pytest.approx(dist, abs=1e-4) == 1.0
        assert changed == ["lr"]

    def test_numerical_distance_int_params(self):
        # |10 - 5| / max(10, 5) = 5/10 = 0.5
        dist, changed = self.m.calculate_config_distance({"depth": 10}, {"depth": 5})
        assert pytest.approx(dist, abs=1e-6) == 0.5
        assert "depth" in changed

    def test_numerical_distance_multiple_params(self):
        a = {"lr": 0.1, "depth": 10}
        b = {"lr": 0.05, "depth": 5}
        dist, changed = self.m.calculate_config_distance(a, b)
        # lr: 0.5,  depth: 0.5  → total 1.0
        assert pytest.approx(dist, abs=1e-6) == 1.0
        assert set(changed) == {"lr", "depth"}

    # --- categorical params -------------------------------------------------

    def test_categorical_same_value_zero_distance(self):
        dist, changed = self.m.calculate_config_distance(
            {"opt": "adam"}, {"opt": "adam"}
        )
        assert dist == 0.0
        assert changed == []

    def test_categorical_different_value_penalty_one(self):
        dist, changed = self.m.calculate_config_distance(
            {"opt": "adam"}, {"opt": "sgd"}
        )
        assert dist == 1.0
        assert changed == ["opt"]

    def test_mixed_numerical_and_categorical(self):
        a = {"lr": 0.1, "opt": "adam", "depth": 5}
        b = {"lr": 0.05, "opt": "sgd",  "depth": 5}
        dist, changed = self.m.calculate_config_distance(a, b)
        # lr: 0.5 + opt: 1.0 = 1.5
        assert pytest.approx(dist, abs=1e-6) == 1.5
        assert set(changed) == {"lr", "opt"}

    # --- missing keys -------------------------------------------------------

    def test_key_missing_in_b_counts_as_change(self):
        dist, changed = self.m.calculate_config_distance(
            {"lr": 0.1, "depth": 5}, {"lr": 0.1}
        )
        # "depth" missing in b → categorical penalty 1.0
        assert dist == 1.0
        assert "depth" in changed

    def test_key_missing_in_a_counts_as_change(self):
        dist, changed = self.m.calculate_config_distance(
            {"lr": 0.1}, {"lr": 0.1, "depth": 5}
        )
        assert dist == 1.0
        assert "depth" in changed

    # --- return type --------------------------------------------------------

    def test_return_types(self):
        dist, changed = self.m.calculate_config_distance({"lr": 0.1}, {"lr": 0.2})
        assert isinstance(dist, float)
        assert isinstance(changed, list)


# ---------------------------------------------------------------------------
# TestDetectOscillation
# ---------------------------------------------------------------------------

class TestDetectOscillation:
    """detect_oscillation — instability score from trace."""

    def setup_method(self):
        self.m = StabilityMetric()

    def test_empty_trace_returns_zero(self):
        assert self.m.detect_oscillation([]) == 0.0

    def test_single_step_no_repeat(self):
        trace = _trace({"lr": 0.1})
        assert self.m.detect_oscillation(trace) == 0.0

    def test_all_unique_configs_zero_instability(self):
        trace = _trace(
            {"lr": 0.1},
            {"lr": 0.05},
            {"lr": 0.01},
        )
        assert self.m.detect_oscillation(trace) == 0.0

    def test_one_repeat_out_of_four(self):
        # Step 2 revisits step 0's config → 1 repeated / 4 = 0.25
        trace = _trace(
            {"lr": 0.1},
            {"lr": 0.05},
            {"lr": 0.1},   # repeat
            {"lr": 0.01},
        )
        assert self.m.detect_oscillation(trace) == pytest.approx(0.25)

    def test_two_repeats_out_of_four(self):
        # Steps 2 and 3 both revisit step 0 → 2/4 = 0.5
        trace = _trace(
            {"lr": 0.1},
            {"lr": 0.05},
            {"lr": 0.1},
            {"lr": 0.1},
        )
        assert self.m.detect_oscillation(trace) == pytest.approx(0.5)

    def test_all_same_config_max_instability(self):
        # Step 0 is first seen; steps 1-3 are all repeats → 3/4 = 0.75
        cfg = {"lr": 0.1}
        trace = _trace(cfg, cfg, cfg, cfg)
        assert self.m.detect_oscillation(trace) == pytest.approx(0.75)

    def test_dict_insertion_order_does_not_matter(self):
        # Same key-value pairs in different insertion order → treated as same config
        a = {"lr": 0.1, "depth": 5}
        b = {"depth": 5, "lr": 0.1}
        trace = _trace(a, b)
        # b is a repeat of a
        assert self.m.detect_oscillation(trace) == pytest.approx(0.5)

    def test_graceful_fallback_without_config_key(self):
        # If a step has no 'config' key the metric should still run without raising
        trace = [{"lr": 0.1}, {"lr": 0.1}]
        score = self.m.detect_oscillation(trace)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_always_in_zero_one_range(self):
        for n_repeats in range(5):
            trace = _trace({"lr": 0.1}) * 5
            score = self.m.detect_oscillation(trace)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestEvaluateTrace
# ---------------------------------------------------------------------------

class TestEvaluateTrace:
    """evaluate_trace — combined churn + instability report."""

    def setup_method(self):
        self.m = StabilityMetric()

    def test_output_keys_present(self):
        trace = _trace({"lr": 0.1}, {"lr": 0.05})
        result = self.m.evaluate_trace(trace)
        assert "instability_score" in result
        assert "average_churn" in result
        assert "total_churn" in result
        assert "total_steps" in result
        assert "churn_steps" in result

    def test_total_steps_matches_trace_length(self):
        trace = _trace({"lr": 0.1}, {"lr": 0.05}, {"lr": 0.01})
        result = self.m.evaluate_trace(trace)
        assert result["total_steps"] == 3

    def test_churn_steps_is_total_minus_one(self):
        trace = _trace({"lr": 0.1}, {"lr": 0.05}, {"lr": 0.01})
        result = self.m.evaluate_trace(trace)
        assert result["churn_steps"] == 2

    def test_single_step_trace(self):
        trace = _trace({"lr": 0.1})
        result = self.m.evaluate_trace(trace)
        assert result["total_steps"] == 1
        assert result["churn_steps"] == 0
        assert result["average_churn"] == 0.0
        assert result["total_churn"] == 0.0
        assert result["instability_score"] == 0.0

    def test_empty_trace(self):
        result = self.m.evaluate_trace([])
        assert result["total_steps"] == 0
        assert result["churn_steps"] == 0
        assert result["average_churn"] == 0.0
        assert result["instability_score"] == 0.0

    def test_no_change_zero_churn(self):
        cfg = {"lr": 0.1, "depth": 5}
        trace = _trace(cfg, cfg, cfg)
        result = self.m.evaluate_trace(trace)
        assert result["average_churn"] == 0.0
        assert result["total_churn"] == 0.0

    def test_instability_score_reflects_repeats(self):
        trace = _trace(
            {"lr": 0.1},
            {"lr": 0.05},
            {"lr": 0.1},   # repeat → 1 repeated / 3 = 0.333
        )
        result = self.m.evaluate_trace(trace)
        assert result["instability_score"] == pytest.approx(1 / 3)

    def test_average_churn_is_mean_of_pairwise_distances(self):
        # Step 0→1: lr 0.1→0.05 → |0.05|/0.1 = 0.5
        # Step 1→2: lr 0.05→0.01 → |0.04|/0.05 = 0.8
        # average = (0.5 + 0.8) / 2 = 0.65
        trace = _trace({"lr": 0.1}, {"lr": 0.05}, {"lr": 0.01})
        result = self.m.evaluate_trace(trace)
        assert result["average_churn"] == pytest.approx(0.65, abs=1e-5)
        assert result["total_churn"] == pytest.approx(1.3, abs=1e-5)

    def test_total_churn_equals_average_times_churn_steps(self):
        trace = _trace(
            {"lr": 0.1, "opt": "adam"},
            {"lr": 0.05, "opt": "sgd"},
            {"lr": 0.025, "opt": "sgd"},
        )
        result = self.m.evaluate_trace(trace)
        if result["churn_steps"] > 0:
            assert result["total_churn"] == pytest.approx(
                result["average_churn"] * result["churn_steps"], abs=1e-9
            )

    def test_output_values_are_numeric(self):
        trace = _trace({"lr": 0.1}, {"lr": 0.05})
        result = self.m.evaluate_trace(trace)
        for key in ("instability_score", "average_churn", "total_churn"):
            assert isinstance(result[key], float), f"{key} should be float"
        for key in ("total_steps", "churn_steps"):
            assert isinstance(result[key], int), f"{key} should be int"

    def test_realistic_benchmark_trace(self):
        """Mirrors a real BaseBenchmark history dict structure."""
        trace = [
            {"step": 0, "config": {"lr": 0.1,  "max_depth": 5,  "optimizer": "adam"}, "metrics": {"accuracy": 0.80}, "proposal_source": "baseline"},
            {"step": 1, "config": {"lr": 0.05, "max_depth": 5,  "optimizer": "adam"}, "metrics": {"accuracy": 0.82}, "proposal_source": "llm"},
            {"step": 2, "config": {"lr": 0.05, "max_depth": 8,  "optimizer": "adam"}, "metrics": {"accuracy": 0.84}, "proposal_source": "llm"},
            {"step": 3, "config": {"lr": 0.1,  "max_depth": 5,  "optimizer": "adam"}, "metrics": {"accuracy": 0.81}, "proposal_source": "llm"},  # revisit step 0
        ]
        result = self.m.evaluate_trace(trace)
        assert result["total_steps"] == 4
        assert result["churn_steps"] == 3
        # Step 3 revisits step 0 → instability = 1/4
        assert result["instability_score"] == pytest.approx(0.25)
        assert result["average_churn"] > 0.0
