"""Forest Cover Type Prediction environment wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.benchmarks.base import BaseEnv


class ForestEnv(BaseEnv):
    """Environment wrapper around the Forest Cover Type workspace."""

    def __init__(self, workspace: Path | None = None):
        super().__init__(workspace)
        # Additional paths specific to Forest Cover
        self.context_path = self.workspace / "dataset_context.json"
        self.features_path = self.workspace / "features.npy"
        self.labels_path = self.workspace / "labels.npy"
        self._validate_required_files()

    def _get_default_workspace(self) -> Path:
        """Return the default workspace path."""
        return Path(__file__).resolve().parent / "workspace"

    def _validate_required_files(self) -> None:
        """Validate that required Forest Cover workspace files exist."""
        for required in (self.features_path, self.labels_path, self.context_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/prepare_forest.py to build the workspace artifacts."
                )
