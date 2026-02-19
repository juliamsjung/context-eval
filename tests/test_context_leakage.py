"""Tests for trace/context boundary enforcement.

These tests verify that trace-only fields cannot leak into agent-visible context.
"""
from __future__ import annotations

import pytest

from src.context import ContextBundle, ContextLeakageError, ContextAxes, ContextBuilder
from src.trace import TRACE_ONLY_FIELDS


class TestTraceOnlyFieldsDefinition:
    """Tests for TRACE_ONLY_FIELDS constant."""

    def test_trace_only_fields_is_frozenset(self):
        """TRACE_ONLY_FIELDS should be immutable."""
        assert isinstance(TRACE_ONLY_FIELDS, frozenset)

    def test_trace_only_fields_contains_expected_fields(self):
        """TRACE_ONLY_FIELDS should contain key observability fields."""
        expected = {
            "token_usage",
            "api_cost",
            "latency_sec",
            "total_api_cost",
            "total_tokens",
            "total_latency_sec",
            "experiment_tags",
            "config_hash",
            "input_tokens",
            "output_tokens",
            "model",
            "temperature",
        }
        assert expected == TRACE_ONLY_FIELDS


class TestContextBundleValidation:
    """Tests for ContextBundle trace leakage detection."""

    def test_valid_bundle_creation(self):
        """Normal bundles should be created without error."""
        bundle = ContextBundle(
            current_config={"learning_rate": 0.1, "max_depth": 5},
            latest_score=0.85,
            recent_history=[
                {"step": 0, "config": {"learning_rate": 0.1}, "score": 0.80},
                {"step": 1, "config": {"learning_rate": 0.15}, "score": 0.83},
            ],
        )
        assert bundle.latest_score == 0.85
        assert len(bundle.recent_history) == 2

    def test_trace_field_in_config_raises_error(self):
        """Trace-only fields in current_config should raise ContextLeakageError."""
        with pytest.raises(ContextLeakageError) as exc_info:
            ContextBundle(
                current_config={"learning_rate": 0.1, "token_usage": {"input": 100}},
                latest_score=0.85,
                recent_history=[],
            )
        assert "token_usage" in str(exc_info.value)
        assert "current_config" in str(exc_info.value)

    def test_trace_field_in_history_raises_error(self):
        """Trace-only fields in recent_history entries should raise ContextLeakageError."""
        with pytest.raises(ContextLeakageError) as exc_info:
            ContextBundle(
                current_config={"learning_rate": 0.1},
                latest_score=0.85,
                recent_history=[
                    {"step": 0, "config": {"learning_rate": 0.1}, "score": 0.80, "api_cost": 0.001},
                ],
            )
        assert "api_cost" in str(exc_info.value)
        assert "recent_history" in str(exc_info.value)

    def test_trace_field_in_nested_history_config_raises_error(self):
        """Trace-only fields in history entry configs should raise ContextLeakageError."""
        with pytest.raises(ContextLeakageError) as exc_info:
            ContextBundle(
                current_config={"learning_rate": 0.1},
                latest_score=0.85,
                recent_history=[
                    {"step": 0, "config": {"learning_rate": 0.1, "latency_sec": 1.5}, "score": 0.80},
                ],
            )
        assert "latency_sec" in str(exc_info.value)
        assert "recent_history[0].config" in str(exc_info.value)

    @pytest.mark.parametrize("trace_field", list(TRACE_ONLY_FIELDS))
    def test_all_trace_fields_detected_in_config(self, trace_field: str):
        """Each trace-only field should be detected in current_config."""
        with pytest.raises(ContextLeakageError):
            ContextBundle(
                current_config={"learning_rate": 0.1, trace_field: "leaked"},
                latest_score=0.85,
                recent_history=[],
            )

    @pytest.mark.parametrize("trace_field", list(TRACE_ONLY_FIELDS))
    def test_all_trace_fields_detected_in_history(self, trace_field: str):
        """Each trace-only field should be detected in history entries."""
        with pytest.raises(ContextLeakageError):
            ContextBundle(
                current_config={"learning_rate": 0.1},
                latest_score=0.85,
                recent_history=[
                    {"step": 0, "config": {}, "score": 0.80, trace_field: "leaked"},
                ],
            )

    def test_bundle_is_frozen(self):
        """ContextBundle should be immutable."""
        bundle = ContextBundle(
            current_config={"learning_rate": 0.1},
            latest_score=0.85,
            recent_history=[],
        )
        with pytest.raises(AttributeError):
            bundle.latest_score = 0.90  # type: ignore

    def test_to_dict_method(self):
        """to_dict should return proper dictionary representation."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[{"step": 0, "config": {}, "score": 0.80}],
            task_description="Test task",
            metric_description="Test metric",
        )
        result = bundle.to_dict()
        assert result["current_config"] == {"lr": 0.1}
        assert result["latest_score"] == 0.85
        assert result["recent_history"] == [{"step": 0, "config": {}, "score": 0.80}]
        assert result["task_description"] == "Test task"
        assert result["metric_description"] == "Test metric"


class TestContextAxes:
    """Tests for ContextAxes configuration."""

    def test_default_axes(self):
        """Default axes should have sensible values."""
        axes = ContextAxes()
        assert axes.feedback_depth == 1
        assert axes.show_task is False
        assert axes.show_metric is False
        assert axes.show_bounds is False

    def test_custom_axes(self):
        """Custom axes should be settable."""
        axes = ContextAxes(feedback_depth=10, show_task=True, show_metric=True, show_bounds=True)
        assert axes.feedback_depth == 10
        assert axes.show_task is True
        assert axes.show_metric is True
        assert axes.show_bounds is True

    def test_zero_feedback_depth_raises_error(self):
        """Zero feedback_depth should raise ValueError."""
        with pytest.raises(ValueError):
            ContextAxes(feedback_depth=0)

    def test_axes_is_frozen(self):
        """ContextAxes should be immutable."""
        axes = ContextAxes()
        with pytest.raises(AttributeError):
            axes.feedback_depth = 10  # type: ignore


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_simple_bundle(self):
        """Builder should create valid bundles."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(feedback_depth=3),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step, accuracy):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": accuracy}
                self.token_usage = None

        # history[-3:-1] gives step 0 (excluding step 1 which is current)
        history = [FakeResult(0, 0.85), FakeResult(1, 0.88)]
        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        assert bundle.current_config == {"lr": 0.2}
        assert bundle.latest_score == 0.90
        assert len(bundle.recent_history) == 1
        assert bundle.recent_history[0]["score"] == 0.85

    def test_build_respects_feedback_depth(self):
        """Builder should respect feedback_depth setting."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(feedback_depth=3),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": 0.85}
                self.token_usage = None

        history = [FakeResult(i) for i in range(5)]
        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        # feedback_depth=3 means current + 2 previous entries
        # history[-3:-1] gives steps 2 and 3 (excluding step 4 which is current)
        assert len(bundle.recent_history) == 2
        assert bundle.recent_history[0]["step"] == 2
        assert bundle.recent_history[1]["step"] == 3

    def test_build_with_feedback_depth_one(self):
        """Builder should exclude history when feedback_depth is 1 (current only)."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(feedback_depth=1),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 0
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.recent_history == []

    def test_feedback_depth_exceeds_history_length(self):
        """Verify graceful handling when feedback_depth > history length."""
        def score_extractor(metrics):
            return metrics.get("score", 0.0)

        axes = ContextAxes(feedback_depth=10)  # Request 10 but only 2 available
        builder = ContextBuilder(axes=axes, score_extractor=score_extractor)

        class MockIterationResult:
            def __init__(self, step, config, metrics):
                self.step = step
                self.config = config
                self.metrics = metrics
                self.token_usage = None

        history = [
            MockIterationResult(step=0, config={"x": 1}, metrics={"score": 0.5}),
            MockIterationResult(step=1, config={"x": 2}, metrics={"score": 0.6}),
        ]
        bundle = builder.build({"x": 3}, {"score": 0.7}, history)
        # Should return 1 history entry (all available minus current)
        assert len(bundle.recent_history) == 1
        assert bundle.recent_history[0]["step"] == 0


class TestBoundsGating:
    """Tests for parameter bounds visibility gating."""

    def test_show_bounds_false_excludes_bounds(self):
        """bounds should be None when show_bounds is False."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        param_bounds = {"lr": (0.01, 0.5), "max_depth": (1, 10)}
        builder = ContextBuilder(
            axes=ContextAxes(show_bounds=False),
            score_extractor=score_extractor,
            param_bounds=param_bounds,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.bounds is None

    def test_show_bounds_true_includes_bounds(self):
        """bounds should be populated when show_bounds is True."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        param_bounds = {"lr": (0.01, 0.5), "max_depth": (1, 10)}
        builder = ContextBuilder(
            axes=ContextAxes(show_bounds=True),
            score_extractor=score_extractor,
            param_bounds=param_bounds,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.bounds is not None
        assert bundle.bounds["lr"] == (0.01, 0.5)
        assert bundle.bounds["max_depth"] == (1, 10)

    def test_show_bounds_true_but_no_param_bounds(self):
        """bounds should be None when show_bounds is True but no param_bounds provided."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_bounds=True),
            score_extractor=score_extractor,
            param_bounds=None,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.bounds is None

    def test_bounds_in_to_dict(self):
        """bounds should appear in to_dict() when present."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            bounds={"lr": (0.01, 0.5), "max_depth": (1, 10)},
        )
        result = bundle.to_dict()
        assert "bounds" in result
        assert result["bounds"]["lr"] == (0.01, 0.5)

    def test_bounds_not_in_to_dict_when_none(self):
        """bounds should not appear in to_dict() when None."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            bounds=None,
        )
        result = bundle.to_dict()
        assert "bounds" not in result
