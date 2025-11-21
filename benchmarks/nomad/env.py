from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


class NomadEnv:
    """Environment wrapper around the NOMAD workspace."""

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(__file__).resolve().parent / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.base_config_path = self.workspace / "config.json"
        self.config_path = self.workspace / "run_config.json"
        self.results_path = self.workspace / "results.json"
        self.context_path = self.workspace / "dataset_context.json"
        self.features_path = self.workspace / "features.npy"
        self.targets_path = self.workspace / "targets.npy"
        self.train_script = self.workspace / "train.py"

        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config missing at {self.base_config_path}")
        if not self.config_path.exists():
            self.config_path.write_text(self.base_config_path.read_text())
        for required in (self.features_path, self.targets_path, self.context_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/prepare_nomad.py to build the workspace artifacts."
                )

    def read_config(self) -> Dict[str, Any]:
        return json.loads(self.config_path.read_text())

    def write_config(self, cfg: Dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(cfg, indent=2))

    def read_context(self) -> Dict[str, Any]:
        if not self.context_path.exists():
            return {}
        return json.loads(self.context_path.read_text())

    def run_train(self) -> Dict[str, Any]:
        subprocess.run(
            [sys.executable, str(self.train_script)],
            cwd=self.workspace,
            check=True,
        )
        return json.loads(self.results_path.read_text())

