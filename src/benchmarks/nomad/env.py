"""NOMAD environment wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.benchmarks.base import BaseEnv


class NomadEnv(BaseEnv):
    """Environment wrapper around the NOMAD workspace."""

    def __init__(self, workspace: Path | None = None):
        super().__init__(workspace)
        # Additional paths specific to NOMAD
        self.context_path = self.workspace / "dataset_context.json"
        self.features_path = self.workspace / "features.npy"
        self.targets_path = self.workspace / "targets.npy"
        self._validate_required_files()

    def _get_default_workspace(self) -> Path:
        """Return the default workspace path."""
        return Path(__file__).resolve().parent / "workspace"

    def _validate_required_files(self) -> None:
        """Validate that required NOMAD workspace files exist."""
        for required in (self.features_path, self.targets_path, self.context_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/prepare_nomad.py to build the workspace artifacts."
                )
