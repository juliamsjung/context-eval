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
        assert axes.history_window == 5
        assert axes.show_task is False
        assert axes.show_metric is False

    def test_custom_axes(self):
        """Custom axes should be settable."""
        axes = ContextAxes(history_window=10, show_task=True, show_metric=True)
        assert axes.history_window == 10
        assert axes.show_task is True
        assert axes.show_metric is True

    def test_negative_history_window_raises_error(self):
        """Negative history_window should raise ValueError."""
        with pytest.raises(ValueError):
            ContextAxes(history_window=-1)

    def test_axes_is_frozen(self):
        """ContextAxes should be immutable."""
        axes = ContextAxes()
        with pytest.raises(AttributeError):
            axes.history_window = 10  # type: ignore


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_simple_bundle(self):
        """Builder should create valid bundles."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(history_window=2),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 0
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.current_config == {"lr": 0.2}
        assert bundle.latest_score == 0.90
        assert len(bundle.recent_history) == 1
        assert bundle.recent_history[0]["score"] == 0.85

    def test_build_respects_history_window(self):
        """Builder should respect history_window setting."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(history_window=2),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": 0.85}

        history = [FakeResult(i) for i in range(5)]
        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        # Should only include last 2 entries
        assert len(bundle.recent_history) == 2
        assert bundle.recent_history[0]["step"] == 3
        assert bundle.recent_history[1]["step"] == 4

    def test_build_with_zero_history_window(self):
        """Builder should exclude history when window is 0."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(history_window=0),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 0
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.recent_history == []
