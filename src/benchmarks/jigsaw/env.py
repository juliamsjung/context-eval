"""Jigsaw Toxic Comment Classification environment wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.benchmarks.base import BaseEnv


class JigsawEnv(BaseEnv):
    """Environment wrapper around the Jigsaw Toxic Comment workspace."""

    # The 6 toxicity label columns
    LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, workspace: Path | None = None):
        super().__init__(workspace)
        # Additional paths specific to Jigsaw
        self.context_path = self.workspace / "dataset_context.json"
        self.texts_path = self.workspace / "texts.json"
        self.ids_path = self.workspace / "ids.json"
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
        """Validate that required Jigsaw workspace files exist."""
        for required in (self.texts_path, self.labels_path, self.context_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/prepare_jigsaw.py to build the workspace artifacts."
                )

    def read_label_mapping(self) -> Dict[str, int]:
        """Read the label mapping (toxicity type -> index)."""
        if not self.label_mapping_path.exists():
            return {}
        return json.loads(self.label_mapping_path.read_text())

    def read_texts(self) -> List[str]:
        """Read the comment texts."""
        return json.loads(self.texts_path.read_text())

    def read_ids(self) -> List[str]:
        """Read the comment IDs."""
        return json.loads(self.ids_path.read_text())
