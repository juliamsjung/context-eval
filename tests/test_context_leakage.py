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
        assert axes.history_window == 0
        assert axes.show_task is False
        assert axes.show_metric is False
        assert axes.show_resources is False

    def test_custom_axes(self):
        """Custom axes should be settable."""
        axes = ContextAxes(history_window=10, show_task=True, show_metric=True, show_resources=True)
        assert axes.history_window == 10
        assert axes.show_task is True
        assert axes.show_metric is True
        assert axes.show_resources is True

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
            token_usage = None

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
                self.token_usage = None

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
            token_usage = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.recent_history == []


class TestResourceSummaryGating:
    """Tests for resource summary visibility gating."""

    def test_show_resources_false_excludes_summary(self):
        """resource_summary should be None when show_resources is False."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=False),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = {
                "total_tokens": 100,
                "api_cost": 0.001,
                "latency_sec": 0.5,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.resource_summary is None

    def test_show_resources_true_includes_summary(self):
        """resource_summary should be populated when show_resources is True."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = {
                "total_tokens": 100,
                "api_cost": 0.001,
                "latency_sec": 0.5,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_used_so_far"] == 100
        assert bundle.resource_summary["api_cost_so_far"] == 0.001
        assert bundle.resource_summary["mean_latency_sec"] == 0.5

    def test_resource_summary_aggregates_multiple_entries(self):
        """resource_summary should aggregate across multiple history entries."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step, tokens, cost, latency):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": 0.85}
                self.token_usage = {
                    "total_tokens": tokens,
                    "api_cost": cost,
                    "latency_sec": latency,
                }

        history = [
            FakeResult(1, 100, 0.001, 0.5),
            FakeResult(2, 150, 0.002, 0.6),
            FakeResult(3, 200, 0.003, 0.7),
        ]

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_used_so_far"] == 450  # 100 + 150 + 200
        assert bundle.resource_summary["api_cost_so_far"] == 0.006  # 0.001 + 0.002 + 0.003
        # Mean latency: (0.5 + 0.6 + 0.7) / 3 = 0.6
        assert bundle.resource_summary["mean_latency_sec"] == 0.6

    def test_resource_summary_handles_missing_token_usage(self):
        """resource_summary should handle entries without token_usage."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=True),
            score_extractor=score_extractor,
        )

        class FakeResultWithUsage:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = {
                "total_tokens": 100,
                "api_cost": 0.001,
                "latency_sec": 0.5,
            }

        class FakeResultWithoutUsage:
            step = 0
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.80}
            token_usage = None  # Baseline has no token usage

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResultWithoutUsage(), FakeResultWithUsage()],
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_used_so_far"] == 100
        assert bundle.resource_summary["api_cost_so_far"] == 0.001

    def test_resource_summary_in_to_dict(self):
        """resource_summary should appear in to_dict() when present."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            resource_summary={
                "tokens_used_so_far": 100,
                "api_cost_so_far": 0.001,
                "mean_latency_sec": 0.5,
            },
        )
        result = bundle.to_dict()
        assert "resource_summary" in result
        assert result["resource_summary"]["tokens_used_so_far"] == 100

    def test_resource_summary_not_in_to_dict_when_none(self):
        """resource_summary should not appear in to_dict() when None."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            resource_summary=None,
        )
        result = bundle.to_dict()
        assert "resource_summary" not in result

    def test_empty_history_returns_zero_resources(self):
        """resource_summary should return zeros for empty history."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=True),
            score_extractor=score_extractor,
        )

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[],
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_used_so_far"] == 0
        assert bundle.resource_summary["api_cost_so_far"] == 0.0
        assert bundle.resource_summary["mean_latency_sec"] == 0.0
