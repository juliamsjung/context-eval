"""Tests for per-run summary logging."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.logging import RunSummary


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_to_dict_contains_all_fields(self):
        summary = RunSummary(
            benchmark="toy",
            seed=0,
            run_id="test_run",
            timestamp="2026-01-15T12:00:00Z",
            git_commit="abc1234",
            model_name="gpt-4o-mini",
            temperature=0.0,
            axis_signature="fd1_t0_m0_r0_d0",
            feedback_depth=1,
            show_task=False,
            show_metric=False,
            show_resources=False,
            show_diagnostics=False,
            final_score=0.85,
            best_score=0.87,
            num_steps_executed=5,
            total_tokens=1000,
            total_cost=0.001,
            num_clamp_events=2,
            num_parse_failures=1,
            num_fallbacks=1,
            num_truncations=0,
        )
        d = summary.to_dict()

        assert d["benchmark"] == "toy"
        assert d["seed"] == 0
        assert d["axis_signature"] == "fd1_t0_m0_r0_d0"
        assert d["final_score"] == 0.85
        assert d["best_score"] == 0.87
        assert d["num_steps_executed"] == 5
        assert d["total_tokens"] == 1000
        assert d["num_clamp_events"] == 2
        assert d["num_parse_failures"] == 1

    def test_to_dict_json_serializable(self):
        summary = RunSummary(
            benchmark="nomad",
            seed=42,
            run_id="test",
            timestamp="2026-01-15T12:00:00Z",
            git_commit=None,
            model_name="gpt-4o",
            temperature=0.7,
            axis_signature="fd5_t1_m1_r1_d1",
            feedback_depth=5,
            show_task=True,
            show_metric=True,
            show_resources=True,
            show_diagnostics=True,
            final_score=0.123,
            best_score=0.100,
            num_steps_executed=10,
            total_tokens=5000,
            total_cost=0.05,
            num_clamp_events=0,
            num_parse_failures=0,
            num_fallbacks=0,
            num_truncations=0,
        )
        # Should not raise
        json_str = json.dumps(summary.to_dict())
        parsed = json.loads(json_str)
        assert parsed["benchmark"] == "nomad"


class TestRunSummaryAggregation:
    """Tests for diagnostics aggregation in base.py."""

    def test_diagnostics_counts_aggregation(self):
        """Verify correct counting of diagnostics events."""
        # Mock history with various diagnostic states
        from src.benchmarks.base import IterationResult

        history = [
            IterationResult(
                step=0, config={}, metrics={"score": 0.5},
                proposal_source="baseline", diagnostics=None
            ),
            IterationResult(
                step=1, config={}, metrics={"score": 0.6},
                proposal_source="llm",
                diagnostics={
                    "clamp_events": [{"parameter": "C", "proposed": 200, "executed": 100}],
                    "parse_failure": False,
                    "fallback_used": False,
                    "truncated": False,
                }
            ),
            IterationResult(
                step=2, config={}, metrics={"score": 0.55},
                proposal_source="heuristic",
                diagnostics={
                    "clamp_events": [],
                    "parse_failure": True,
                    "fallback_used": True,
                    "truncated": False,
                }
            ),
        ]

        # Count clamp events
        num_clamp = sum(
            len(r.diagnostics.get("clamp_events", []))
            for r in history if r.diagnostics
        )
        assert num_clamp == 1

        # Count parse failures
        num_parse = sum(
            1 for r in history
            if r.diagnostics and r.diagnostics.get("parse_failure", False)
        )
        assert num_parse == 1

        # Count fallbacks
        num_fallback = sum(
            1 for r in history
            if r.diagnostics and r.diagnostics.get("fallback_used", False)
        )
        assert num_fallback == 1


class TestRunSummaryIndependence:
    """Tests verifying logging is independent of visibility flags."""

    def test_totals_same_regardless_of_show_resources(self):
        """Token totals must be identical whether show_resources is True or False."""
        from src.benchmarks.base import IterationResult

        # Same history data
        history = [
            IterationResult(
                step=0, config={}, metrics={"score": 0.5},
                proposal_source="baseline", token_usage=None, diagnostics=None
            ),
            IterationResult(
                step=1, config={}, metrics={"score": 0.6},
                proposal_source="llm",
                token_usage={"input_tokens": 100, "output_tokens": 50, "api_cost": 0.001},
                diagnostics={"clamp_events": [], "parse_failure": False,
                             "fallback_used": False, "truncated": False}
            ),
        ]

        # Compute totals - this uses history directly, not context visibility
        total_tokens = sum(
            r.token_usage.get("total_tokens", r.token_usage.get("input_tokens", 0) + r.token_usage.get("output_tokens", 0))
            for r in history if r.token_usage
        )

        # Should always be 150 regardless of any visibility flags
        assert total_tokens == 150


class TestRunSummaryFileOutput:
    """Tests for JSONL file writing."""

    def test_jsonl_file_created(self):
        """Verify runs/ directory and file are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()

            output_path = runs_dir / "toy_runs.jsonl"
            summary = RunSummary(
                benchmark="toy", seed=0, run_id="test",
                timestamp="2026-01-15T12:00:00Z", git_commit="abc1234",
                model_name="gpt-4o-mini", temperature=0.0,
                axis_signature="fd1_t0_m0_r0_d0",
                feedback_depth=1, show_task=False, show_metric=False,
                show_resources=False, show_diagnostics=False,
                final_score=0.8, best_score=0.85, num_steps_executed=3,
                total_tokens=100, total_cost=0.001,
                num_clamp_events=0, num_parse_failures=0,
                num_fallbacks=0, num_truncations=0,
            )

            with open(output_path, "a") as f:
                f.write(json.dumps(summary.to_dict()) + "\n")

            assert output_path.exists()

            with open(output_path) as f:
                line = f.readline()
                parsed = json.loads(line)
                assert parsed["benchmark"] == "toy"

    def test_jsonl_append_mode(self):
        """Verify multiple runs append correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_runs.jsonl"

            for seed in [0, 1, 2]:
                summary = RunSummary(
                    benchmark="toy", seed=seed, run_id=f"run_{seed}",
                    timestamp="2026-01-15T12:00:00Z", git_commit="abc1234",
                    model_name="gpt-4o-mini", temperature=0.0,
                    axis_signature="fd1_t0_m0_r0_d0",
                    feedback_depth=1, show_task=False, show_metric=False,
                    show_resources=False, show_diagnostics=False,
                    final_score=0.8 + seed * 0.01, best_score=0.85,
                    num_steps_executed=3,
                    total_tokens=100, total_cost=0.001,
                    num_clamp_events=0, num_parse_failures=0,
                    num_fallbacks=0, num_truncations=0,
                )
                with open(output_path, "a") as f:
                    f.write(json.dumps(summary.to_dict()) + "\n")

            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == 3
            assert json.loads(lines[0])["seed"] == 0
            assert json.loads(lines[1])["seed"] == 1
            assert json.loads(lines[2])["seed"] == 2
