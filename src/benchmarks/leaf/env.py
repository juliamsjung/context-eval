"""Leaf Classification environment wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.benchmarks.base import BaseEnv


class LeafEnv(BaseEnv):
    """Environment wrapper around the Leaf Classification workspace."""

    def __init__(self, workspace: Path | None = None):
        super().__init__(workspace)
        # Additional paths specific to Leaf
        self.context_path = self.workspace / "dataset_context.json"
        self.features_path = self.workspace / "features.npy"
        self.labels_path = self.workspace / "labels.npy"
        self.label_mapping_path = self.workspace / "label_mapping.json"
        self._validate_required_files()

    def _get_default_workspace(self) -> Path:
        """Return the default workspace path."""
        return Path(__file__).resolve().parent / "workspace"

    def _init_config(self) -> None:
        """Initialize run_config.json from base config (always reset to ensure clean baseline)."""
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config missing at {self.base_config_path}")
        self.config_path.write_text(self.base_config_path.read_text())

    def _validate_required_files(self) -> None:
        """Validate that required Leaf workspace files exist."""
        for required in (self.features_path, self.labels_path, self.context_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/prepare_leaf.py to build the workspace artifacts."
                )

    def read_label_mapping(self) -> Dict[str, int]:
        """Read the label mapping (species name -> index)."""
        if not self.label_mapping_path.exists():
            return {}
        return json.loads(self.label_mapping_path.read_text())
