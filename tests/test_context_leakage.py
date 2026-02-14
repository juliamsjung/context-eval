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
        assert axes.show_resources is False
        assert axes.show_diagnostics is False

    def test_custom_axes(self):
        """Custom axes should be settable."""
        axes = ContextAxes(feedback_depth=10, show_task=True, show_metric=True, show_resources=True, show_diagnostics=True)
        assert axes.feedback_depth == 10
        assert axes.show_task is True
        assert axes.show_metric is True
        assert axes.show_resources is True
        assert axes.show_diagnostics is True

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
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_current"] == 100
        assert bundle.resource_summary["tokens_cumulative"] == 100
        assert bundle.resource_summary["cost_cumulative"] == 0.001

    def test_resource_summary_aggregates_multiple_entries(self):
        """resource_summary should aggregate across multiple history entries."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_resources=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step, tokens, cost):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": 0.85}
                self.token_usage = {
                    "total_tokens": tokens,
                    "api_cost": cost,
                }

        history = [
            FakeResult(1, 100, 0.001),
            FakeResult(2, 150, 0.002),
            FakeResult(3, 200, 0.003),
        ]

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        assert bundle.resource_summary is not None
        assert bundle.resource_summary["tokens_current"] == 200  # Last entry's tokens
        assert bundle.resource_summary["tokens_cumulative"] == 450  # 100 + 150 + 200
        assert bundle.resource_summary["cost_cumulative"] == 0.006  # 0.001 + 0.002 + 0.003

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
        assert bundle.resource_summary["tokens_current"] == 100
        assert bundle.resource_summary["tokens_cumulative"] == 100
        assert bundle.resource_summary["cost_cumulative"] == 0.001

    def test_resource_summary_in_to_dict(self):
        """resource_summary should appear in to_dict() when present."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            resource_summary={
                "tokens_current": 100,
                "tokens_cumulative": 100,
                "cost_cumulative": 0.001,
            },
        )
        result = bundle.to_dict()
        assert "resource_summary" in result
        assert result["resource_summary"]["tokens_cumulative"] == 100

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
        assert bundle.resource_summary["tokens_current"] == 0
        assert bundle.resource_summary["tokens_cumulative"] == 0
        assert bundle.resource_summary["cost_cumulative"] == 0.0


class TestDiagnosticsGating:
    """Tests for diagnostics visibility gating."""

    def test_show_diagnostics_false_excludes_diagnostics(self):
        """diagnostics should be None when show_diagnostics is False."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=False),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None
            diagnostics = {
                "clamp_events": [],
                "parse_failure": False,
                "fallback_used": False,
                "truncated": False,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.diagnostics is None

    def test_show_diagnostics_true_includes_diagnostics(self):
        """diagnostics should be populated when show_diagnostics is True."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None
            diagnostics = {
                "clamp_events": [],
                "parse_failure": False,
                "fallback_used": False,
                "truncated": False,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.diagnostics is not None
        assert bundle.diagnostics["clamp_events"] == []
        assert bundle.diagnostics["parse_failure"] is False
        assert bundle.diagnostics["fallback_used"] is False
        assert bundle.diagnostics["truncated"] is False

    def test_diagnostics_clamp_detection(self):
        """diagnostics should include clamp events when values were clamped."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None
            diagnostics = {
                "clamp_events": [
                    {"parameter": "max_iter", "proposed": 1200, "executed": 1000},
                    {"parameter": "learning_rate", "proposed": 0.8, "executed": 0.5},
                ],
                "parse_failure": False,
                "fallback_used": False,
                "truncated": False,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.diagnostics is not None
        assert len(bundle.diagnostics["clamp_events"]) == 2
        assert bundle.diagnostics["clamp_events"][0]["parameter"] == "max_iter"
        assert bundle.diagnostics["clamp_events"][0]["proposed"] == 1200
        assert bundle.diagnostics["clamp_events"][0]["executed"] == 1000

    def test_diagnostics_truncation_detection(self):
        """diagnostics should indicate truncation when LLM response was truncated."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = {"finish_reason": "length"}
            diagnostics = {
                "clamp_events": [],
                "parse_failure": True,
                "fallback_used": True,
                "truncated": True,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.diagnostics is not None
        assert bundle.diagnostics["truncated"] is True
        assert bundle.diagnostics["parse_failure"] is True

    def test_diagnostics_fallback_detection(self):
        """diagnostics should indicate fallback when heuristic was used."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None
            diagnostics = {
                "clamp_events": [],
                "parse_failure": False,
                "fallback_used": True,
                "truncated": False,
            }

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResult()],
        )

        assert bundle.diagnostics is not None
        assert bundle.diagnostics["fallback_used"] is True
        assert bundle.diagnostics["parse_failure"] is False

    def test_diagnostics_only_current_step(self):
        """diagnostics should only show data from the most recent step."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResult:
            def __init__(self, step, clamp_events):
                self.step = step
                self.config = {"lr": 0.1}
                self.metrics = {"accuracy": 0.85}
                self.token_usage = None
                self.diagnostics = {
                    "clamp_events": clamp_events,
                    "parse_failure": False,
                    "fallback_used": False,
                    "truncated": False,
                }

        # Step 1 had clamp events, step 2 did not
        history = [
            FakeResult(1, [{"parameter": "C", "proposed": 200, "executed": 100}]),
            FakeResult(2, []),
        ]

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=history,
        )

        # Should only get diagnostics from step 2 (most recent)
        assert bundle.diagnostics is not None
        assert bundle.diagnostics["clamp_events"] == []

    def test_diagnostics_empty_history_returns_none(self):
        """diagnostics should be None when history is empty."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[],
        )

        assert bundle.diagnostics is None

    def test_diagnostics_in_to_dict(self):
        """diagnostics should appear in to_dict() when present."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            diagnostics={
                "clamp_events": [],
                "parse_failure": False,
                "fallback_used": False,
                "truncated": False,
            },
        )
        result = bundle.to_dict()
        assert "diagnostics" in result
        assert result["diagnostics"]["parse_failure"] is False

    def test_diagnostics_not_in_to_dict_when_none(self):
        """diagnostics should not appear in to_dict() when None."""
        bundle = ContextBundle(
            current_config={"lr": 0.1},
            latest_score=0.85,
            recent_history=[],
            diagnostics=None,
        )
        result = bundle.to_dict()
        assert "diagnostics" not in result

    def test_diagnostics_no_diagnostics_field_in_history(self):
        """diagnostics should be None when history entries lack diagnostics field."""
        def score_extractor(metrics):
            return metrics.get("accuracy", 0.0)

        builder = ContextBuilder(
            axes=ContextAxes(show_diagnostics=True),
            score_extractor=score_extractor,
        )

        class FakeResultWithoutDiagnostics:
            step = 1
            config = {"lr": 0.1}
            metrics = {"accuracy": 0.85}
            token_usage = None
            diagnostics = None

        bundle = builder.build(
            current_config={"lr": 0.2},
            last_metrics={"accuracy": 0.90},
            history=[FakeResultWithoutDiagnostics()],
        )

        assert bundle.diagnostics is None
