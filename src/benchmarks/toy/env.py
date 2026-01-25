"""Toy tabular environment wrapper."""
from __future__ import annotations

from pathlib import Path

from src.benchmarks.base import BaseEnv


class ToyTabularEnv(BaseEnv):
    """Minimal environment wrapper around the toy tabular workspace."""

    def _get_default_workspace(self) -> Path:
        """Return the default workspace path."""
        return Path(__file__).resolve().parent / "workspace"

    def _init_config(self) -> None:
        """Initialize run_config.json from base config if it doesn't exist."""
        if not self.config_path.exists():
            if not self.base_config_path.exists():
                raise FileNotFoundError(
                    f"Baseline config.json not found at {self.base_config_path}"
                )
            self.config_path.write_text(self.base_config_path.read_text())
