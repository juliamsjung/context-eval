"""Tests for landscape characterization module.

Covers SobolSampler, LandscapeRunner, StratifiedSelector, and regression
tests ensuring existing benchmark behavior is unchanged.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest

from src.landscape.sampler import SobolSampler
from src.landscape.runner import LandscapeRunner
from src.landscape.selector import StratifiedSelector


# ---------------------------------------------------------------------------
# Test PARAM_BOUNDS constants
# ---------------------------------------------------------------------------

# Old bounds for regression checking (new bounds must be supersets)
OLD_NOMAD_BOUNDS = {
    "learning_rate": (0.01, 0.5),
    "max_depth": (2, 16),
    "max_iter": (50, 2000),
    "l2_regularization": (0.0, 2.0),
    "max_leaf_nodes": (15, 255),
    "min_samples_leaf": (1, 200),
}
OLD_JIGSAW_BOUNDS = {
    "max_features": (1000, 50000),
    "ngram_max": (1, 3),
    "min_df": (1, 20),
    "C": (0.01, 10.0),
    "max_iter": (50, 1000),
}
OLD_FOREST_BOUNDS = {
    "n_estimators": (50, 1000),
    "max_depth": (3, 100),
    "min_samples_split": (2, 50),
    "min_samples_leaf": (1, 30),
    "max_features": (0.1, 1.0),
}


class TestWidenedBoundsRegression:
    """Verify widened bounds are strict supersets of old bounds."""

    def test_nomad_bounds_contain_old(self):
        from src.benchmarks.nomad.benchmark import PARAM_BOUNDS
        # l2_regularization intentionally excluded: old lower bound was 0.0,
        # raised to 1e-4 because log-scale sampling requires positive bounds.
        for key, (old_lo, old_hi) in OLD_NOMAD_BOUNDS.items():
            if key == "l2_regularization":
                continue  # tested separately below
            new_lo, new_hi = PARAM_BOUNDS[key]
            assert new_lo <= old_lo, f"NOMAD {key}: new lower {new_lo} > old {old_lo}"
            assert new_hi >= old_hi, f"NOMAD {key}: new upper {new_hi} < old {old_hi}"

    def test_nomad_l2_regularization_positive_for_log_scale(self):
        """l2_regularization lower bound raised from 0.0 to 1e-4 for log-scale."""
        from src.benchmarks.nomad.benchmark import PARAM_BOUNDS
        lo, hi = PARAM_BOUNDS["l2_regularization"]
        assert lo > 0, "l2_regularization must have positive lower bound for log-scale"
        assert hi >= OLD_NOMAD_BOUNDS["l2_regularization"][1]

    def test_jigsaw_bounds_contain_old(self):
        from src.benchmarks.jigsaw.benchmark import PARAM_BOUNDS
        for key, (old_lo, old_hi) in OLD_JIGSAW_BOUNDS.items():
            new_lo, new_hi = PARAM_BOUNDS[key]
            assert new_lo <= old_lo, f"Jigsaw {key}: new lower {new_lo} > old {old_lo}"
            assert new_hi >= old_hi, f"Jigsaw {key}: new upper {new_hi} < old {old_hi}"

    def test_forest_bounds_contain_old(self):
        from src.benchmarks.forest.benchmark import PARAM_BOUNDS
        for key, (old_lo, old_hi) in OLD_FOREST_BOUNDS.items():
            new_lo, new_hi = PARAM_BOUNDS[key]
            assert new_lo <= old_lo, f"Forest {key}: new lower {new_lo} > old {old_lo}"
            assert new_hi >= old_hi, f"Forest {key}: new upper {new_hi} < old {old_hi}"


class TestBenchmarkConstants:
    """Verify LOG_SCALE_PARAMS and INTEGER_KEYS are consistent."""

    @pytest.mark.parametrize("benchmark_module,expected_log,expected_int", [
        (
            "src.benchmarks.nomad.benchmark",
            {"learning_rate", "l2_regularization"},
            {"max_depth", "max_leaf_nodes", "max_iter", "min_samples_leaf"},
        ),
        (
            "src.benchmarks.jigsaw.benchmark",
            {"C", "max_features"},
            {"max_features", "ngram_max", "min_df", "max_iter"},
        ),
        (
            "src.benchmarks.forest.benchmark",
            set(),
            {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"},
        ),
        (
            "src.benchmarks.housing.benchmark",
            set(),
            {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap"},
        ),
    ])
    def test_constants_exist(self, benchmark_module, expected_log, expected_int):
        import importlib
        mod = importlib.import_module(benchmark_module)
        assert mod.LOG_SCALE_PARAMS == expected_log
        assert mod.INTEGER_KEYS == expected_int

    @pytest.mark.parametrize("benchmark_module", [
        "src.benchmarks.nomad.benchmark",
        "src.benchmarks.jigsaw.benchmark",
        "src.benchmarks.forest.benchmark",
        "src.benchmarks.housing.benchmark",
    ])
    def test_log_scale_params_subset_of_bounds(self, benchmark_module):
        """LOG_SCALE_PARAMS must be a subset of PARAM_BOUNDS keys."""
        import importlib
        mod = importlib.import_module(benchmark_module)
        assert mod.LOG_SCALE_PARAMS <= set(mod.PARAM_BOUNDS.keys())

    @pytest.mark.parametrize("benchmark_module", [
        "src.benchmarks.nomad.benchmark",
        "src.benchmarks.jigsaw.benchmark",
        "src.benchmarks.forest.benchmark",
        "src.benchmarks.housing.benchmark",
    ])
    def test_integer_keys_subset_of_bounds(self, benchmark_module):
        """INTEGER_KEYS must be a subset of PARAM_BOUNDS keys."""
        import importlib
        mod = importlib.import_module(benchmark_module)
        assert mod.INTEGER_KEYS <= set(mod.PARAM_BOUNDS.keys())

    @pytest.mark.parametrize("benchmark_module", [
        "src.benchmarks.nomad.benchmark",
        "src.benchmarks.jigsaw.benchmark",
        "src.benchmarks.forest.benchmark",
        "src.benchmarks.housing.benchmark",
    ])
    def test_log_scale_params_have_positive_bounds(self, benchmark_module):
        """Log-scale params must have strictly positive lower bounds."""
        import importlib
        mod = importlib.import_module(benchmark_module)
        for name in mod.LOG_SCALE_PARAMS:
            lo, hi = mod.PARAM_BOUNDS[name]
            assert lo > 0, f"{name}: lower bound {lo} must be > 0 for log-scale"


# ---------------------------------------------------------------------------
# SobolSampler tests
# ---------------------------------------------------------------------------

SIMPLE_BOUNDS = {
    "x": (0.0, 10.0),
    "y": (1.0, 100.0),
}


class TestSobolSampler:
    """Tests for SobolSampler."""

    def test_sample_count(self):
        """Should return exactly n configs."""
        sampler = SobolSampler(param_bounds=SIMPLE_BOUNDS)
        configs = sampler.sample(n=200)
        assert len(configs) == 200

    def test_sample_count_small(self):
        """Should work with small n values."""
        sampler = SobolSampler(param_bounds=SIMPLE_BOUNDS)
        configs = sampler.sample(n=5)
        assert len(configs) == 5

    def test_all_values_within_bounds(self):
        """Every sampled value must fall within PARAM_BOUNDS."""
        sampler = SobolSampler(param_bounds=SIMPLE_BOUNDS)
        configs = sampler.sample(n=100)
        for config in configs:
            for name, (lo, hi) in SIMPLE_BOUNDS.items():
                assert lo <= config[name] <= hi, (
                    f"{name}={config[name]} outside [{lo}, {hi}]"
                )

    def test_all_keys_present(self):
        """Each config must have all parameter keys."""
        sampler = SobolSampler(param_bounds=SIMPLE_BOUNDS)
        configs = sampler.sample(n=10)
        for config in configs:
            assert set(config.keys()) == set(SIMPLE_BOUNDS.keys())

    def test_log_scale_distribution(self):
        """Log-scale params should be roughly uniform in log-space."""
        bounds = {"lr": (0.001, 1.0)}
        sampler = SobolSampler(
            param_bounds=bounds,
            log_scale_params={"lr"},
            seed=42,
        )
        configs = sampler.sample(n=256)
        values = [c["lr"] for c in configs]

        # In uniform log-space, median should be near geometric mean
        geometric_mean = math.sqrt(0.001 * 1.0)  # ~0.0316
        median = sorted(values)[len(values) // 2]

        # Allow generous tolerance but verify it's log-distributed
        assert median < 0.1, f"Median {median} too high for log-scale (expected ~{geometric_mean})"

    def test_integer_keys(self):
        """Integer params should be int type."""
        bounds = {"n": (1.0, 100.0), "x": (0.0, 1.0)}
        sampler = SobolSampler(
            param_bounds=bounds,
            integer_keys={"n"},
        )
        configs = sampler.sample(n=50)
        for config in configs:
            assert isinstance(config["n"], int), f"Expected int, got {type(config['n'])}"
            assert isinstance(config["x"], float), f"Expected float, got {type(config['x'])}"

    def test_reproducibility(self):
        """Same seed should produce identical samples."""
        s1 = SobolSampler(param_bounds=SIMPLE_BOUNDS, seed=42)
        s2 = SobolSampler(param_bounds=SIMPLE_BOUNDS, seed=42)
        assert s1.sample(20) == s2.sample(20)

    def test_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        s1 = SobolSampler(param_bounds=SIMPLE_BOUNDS, seed=0)
        s2 = SobolSampler(param_bounds=SIMPLE_BOUNDS, seed=1)
        assert s1.sample(20) != s2.sample(20)

    def test_log_scale_positive_bound_required(self):
        """Log-scale with non-positive lower bound should raise ValueError."""
        with pytest.raises(ValueError, match="non-positive"):
            SobolSampler(
                param_bounds={"x": (0.0, 1.0)},
                log_scale_params={"x"},
            )

    def test_real_benchmark_bounds(self):
        """Sampling with actual NOMAD bounds should produce valid configs."""
        from src.benchmarks.nomad.benchmark import PARAM_BOUNDS, LOG_SCALE_PARAMS, INTEGER_KEYS
        sampler = SobolSampler(
            param_bounds=PARAM_BOUNDS,
            log_scale_params=LOG_SCALE_PARAMS,
            integer_keys=INTEGER_KEYS,
        )
        configs = sampler.sample(n=50)
        assert len(configs) == 50
        for config in configs:
            for name, (lo, hi) in PARAM_BOUNDS.items():
                assert lo <= config[name] <= hi, f"{name}={config[name]} outside [{lo}, {hi}]"
            for name in INTEGER_KEYS:
                assert isinstance(config[name], int)


# ---------------------------------------------------------------------------
# LandscapeRunner tests
# ---------------------------------------------------------------------------

class TestLandscapeRunner:
    """Tests for LandscapeRunner."""

    def _make_runner(self) -> LandscapeRunner:
        """Create a runner with a simple mock training function."""
        def mock_train(config):
            return {"score": config["x"] ** 2}

        def mock_extract(metrics):
            return metrics["score"]

        return LandscapeRunner(
            run_training=mock_train,
            score_extractor=mock_extract,
            higher_is_better=True,
            benchmark_name="test",
        )

    def test_evaluate_returns_correct_count(self):
        configs = [{"x": i} for i in range(10)]
        runner = self._make_runner()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = runner.evaluate(
                configs,
                output_path=Path(tmpdir) / "results.json",
            )
        assert len(results) == 10

    def test_evaluate_records_scores(self):
        configs = [{"x": 3.0}]
        runner = self._make_runner()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = runner.evaluate(
                configs,
                output_path=Path(tmpdir) / "results.json",
            )
        assert results[0]["primary_score"] == 9.0

    def test_evaluate_handles_errors(self):
        """Configs that cause errors should be recorded, not crash."""
        def failing_train(config):
            if config["x"] == 1:
                raise ValueError("bad config")
            return {"score": config["x"]}

        runner = LandscapeRunner(
            run_training=failing_train,
            score_extractor=lambda m: m["score"],
            higher_is_better=True,
            benchmark_name="test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            results = runner.evaluate(
                [{"x": 0}, {"x": 1}, {"x": 2}],
                output_path=Path(tmpdir) / "results.json",
            )
        assert len(results) == 3
        assert results[1]["error"] is not None
        assert results[0]["error"] is None

    def test_evaluate_saves_json(self):
        configs = [{"x": 1.0}]
        runner = self._make_runner()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.json"
            runner.evaluate(configs, output_path=output)
            assert output.exists()
            data = json.loads(output.read_text())
            assert data["benchmark"] == "test"
            assert data["num_samples"] == 1


# ---------------------------------------------------------------------------
# StratifiedSelector tests
# ---------------------------------------------------------------------------

class TestStratifiedSelector:
    """Tests for StratifiedSelector."""

    def _make_results(self, n: int = 100, higher_is_better: bool = True) -> List[Dict[str, Any]]:
        """Create mock results with linearly spaced scores."""
        return [
            {
                "index": i,
                "config": {"x": float(i)},
                "metrics": {"score": float(i)},
                "primary_score": float(i),
                "error": None,
            }
            for i in range(n)
        ]

    def test_returns_three_keys(self):
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(100)
        with tempfile.TemporaryDirectory() as tmpdir:
            selections = selector.select(results, output_dir=Path(tmpdir))
        assert set(selections.keys()) == {"low", "neutral", "high"}

    def test_ordering_higher_is_better(self):
        """low.score < neutral.score < high.score for higher_is_better."""
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = selector.select(results, output_dir=Path(tmpdir))
        assert sel["low"]["primary_score"] < sel["neutral"]["primary_score"]
        assert sel["neutral"]["primary_score"] < sel["high"]["primary_score"]

    def test_ordering_lower_is_better(self):
        """low.score > neutral.score > high.score for lower_is_better (reversed)."""
        selector = StratifiedSelector(higher_is_better=False)
        results = self._make_results(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = selector.select(results, output_dir=Path(tmpdir))
        # For lower_is_better, "low quality" means high score (bad)
        assert sel["low"]["primary_score"] > sel["neutral"]["primary_score"]
        assert sel["neutral"]["primary_score"] > sel["high"]["primary_score"]

    def test_high_excludes_top_outliers(self):
        """High config should not be from the top 5%."""
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = selector.select(results, output_dir=Path(tmpdir))
        # Max score is 199, top 5% starts at ~190
        assert sel["high"]["primary_score"] < 190

    def test_saves_individual_configs(self):
        """Should save low.json, neutral.json, high.json files."""
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "configs"
            selector.select(results, output_dir=output_dir)
            for quality in ["low", "neutral", "high"]:
                cfg_path = output_dir / f"{quality}.json"
                assert cfg_path.exists(), f"Missing {quality}.json"
                data = json.loads(cfg_path.read_text())
                assert "x" in data  # Should be a raw config dict

    def test_saves_metadata(self):
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "configs"
            selector.select(results, output_dir=output_dir)
            meta_path = output_dir / "selection_metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert "low" in meta
            assert "primary_score" in meta["low"]

    def test_too_few_results_raises(self):
        """Need at least 4 successful results."""
        selector = StratifiedSelector()
        results = [
            {"config": {"x": 1}, "primary_score": 1.0, "error": None}
        ]
        with pytest.raises(ValueError, match="at least 4"):
            selector.select(results)

    def test_filters_failures(self):
        """Failed evaluation results should be filtered out."""
        selector = StratifiedSelector(higher_is_better=True)
        results = self._make_results(50)
        # Add some failures
        results.append({"index": 99, "config": {}, "metrics": {}, "primary_score": None, "error": "fail"})
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = selector.select(results, output_dir=Path(tmpdir))
        # Should still work and only use the 50 valid results
        assert set(sel.keys()) == {"low", "neutral", "high"}
