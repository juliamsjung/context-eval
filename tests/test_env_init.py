"""Tests for environment config initialization.

Verifies that BaseEnv._init_config properly handles:
1. Default: config.json → run_config.json
2. Override: init_override.json takes precedence over config.json
3. Error: missing config.json with no override raises FileNotFoundError
4. Cleanup: override removal restores default behavior

These tests ensure the stratified init configs from the landscape
characterization pipeline are correctly applied during grid experiments.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.benchmarks.base import BaseEnv


class ConcreteEnv(BaseEnv):
    """Minimal concrete implementation for testing."""

    def __init__(self, workspace: Path):
        # Don't call _validate_required_files (no data files in test)
        super().__init__(workspace)

    def _get_default_workspace(self) -> Path:
        return Path("/nonexistent")


class TestInitConfigDefault:
    """Test default config.json → run_config.json initialization."""

    def test_config_json_copied_to_run_config(self):
        """When only config.json exists, run_config.json should match it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            base_config = {"learning_rate": 0.1, "max_depth": 5}
            (workspace / "config.json").write_text(json.dumps(base_config))
            # Provide a dummy train.py so BaseEnv doesn't fail
            (workspace / "train.py").write_text("# dummy")

            env = ConcreteEnv(workspace)
            run_config = json.loads(env.config_path.read_text())
            assert run_config == base_config

    def test_run_config_is_copy_not_reference(self):
        """Modifying run_config.json should not affect config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            base_config = {"lr": 0.1}
            (workspace / "config.json").write_text(json.dumps(base_config))
            (workspace / "train.py").write_text("# dummy")

            env = ConcreteEnv(workspace)
            # Mutate run_config
            env.write_config({"lr": 0.5})

            # config.json should be unchanged
            original = json.loads((workspace / "config.json").read_text())
            assert original == base_config


class TestInitConfigOverride:
    """Test init_override.json takes precedence over config.json."""

    def test_override_takes_precedence(self):
        """When init_override.json exists, run_config.json should use it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            base_config = {"learning_rate": 0.1, "max_depth": 5}
            override_config = {"learning_rate": 0.01, "max_depth": 15}

            (workspace / "config.json").write_text(json.dumps(base_config))
            (workspace / "init_override.json").write_text(json.dumps(override_config))
            (workspace / "train.py").write_text("# dummy")

            env = ConcreteEnv(workspace)
            run_config = json.loads(env.config_path.read_text())
            assert run_config == override_config
            assert run_config != base_config

    def test_override_with_stratified_config(self):
        """Simulate grid script injecting a landscape-generated config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Default config (tracked in git)
            default = {"n_estimators": 100, "max_depth": 10}
            # Stratified "high quality" init from landscape characterization
            high_quality = {"n_estimators": 500, "max_depth": 25}

            (workspace / "config.json").write_text(json.dumps(default))
            (workspace / "init_override.json").write_text(json.dumps(high_quality))
            (workspace / "train.py").write_text("# dummy")

            env = ConcreteEnv(workspace)
            run_config = env.read_config()

            # Should start from the high-quality landscape config, not default
            assert run_config["n_estimators"] == 500
            assert run_config["max_depth"] == 25

    def test_override_removal_restores_default(self):
        """After removing init_override.json, next init uses config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            base = {"lr": 0.1}
            override = {"lr": 0.5}

            (workspace / "config.json").write_text(json.dumps(base))
            (workspace / "init_override.json").write_text(json.dumps(override))
            (workspace / "train.py").write_text("# dummy")

            # First init: override takes precedence
            env1 = ConcreteEnv(workspace)
            assert env1.read_config() == override

            # Remove override (like grid cleanup does)
            (workspace / "init_override.json").unlink()

            # Second init: falls back to config.json
            env2 = ConcreteEnv(workspace)
            assert env2.read_config() == base


class TestInitConfigMissing:
    """Test error handling when no config is available."""

    def test_missing_config_raises_error(self):
        """Missing config.json with no override should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "train.py").write_text("# dummy")
            # No config.json, no init_override.json

            with pytest.raises(FileNotFoundError, match="Base config missing"):
                ConcreteEnv(workspace)

    def test_override_alone_works_without_config(self):
        """init_override.json should work even if config.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            override = {"lr": 0.2}
            (workspace / "init_override.json").write_text(json.dumps(override))
            (workspace / "train.py").write_text("# dummy")

            env = ConcreteEnv(workspace)
            assert env.read_config() == override


class TestRealBenchmarkEnvsUseBaseInit:
    """Verify that real benchmark Envs inherit BaseEnv._init_config (not override it)."""

    @pytest.mark.parametrize("env_module,env_class_name", [
        ("src.benchmarks.nomad.env", "NomadEnv"),
        ("src.benchmarks.jigsaw.env", "JigsawEnv"),
        ("src.benchmarks.forest.env", "ForestEnv"),
        ("src.benchmarks.housing.env", "HousingEnv"),
    ])
    def test_no_init_config_override(self, env_module, env_class_name):
        """Env classes should NOT override _init_config (base handles it)."""
        import importlib
        mod = importlib.import_module(env_module)
        env_cls = getattr(mod, env_class_name)

        # _init_config should NOT be defined directly on the subclass
        assert "_init_config" not in env_cls.__dict__, (
            f"{env_class_name} overrides _init_config — "
            f"this bypasses init_override.json support from BaseEnv. "
            f"Remove the override to use the base class implementation."
        )
