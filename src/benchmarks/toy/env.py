"""Toy tabular environment wrapper."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


class ToyTabularEnv:
    """Minimal environment wrapper around the toy tabular workspace."""

    def __init__(self, workspace: Path | None = None):
        # Use the old toy_bench location for now for backwards compatibility
        default_workspace = Path(__file__).resolve().parent / "workspace"
        self.workspace = workspace or default_workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Baseline config (tracked in git)
        self.base_config_path = self.workspace / "config.json"
        # Runtime config (mutated, ignored by git)
        self.config_path = self.workspace / "run_config.json"

        self.results_path = self.workspace / "results.json"
        self.train_script = self.workspace / "train.py"

        # On first run, if run_config.json doesn't exist, copy from config.json
        if not self.config_path.exists():
            if not self.base_config_path.exists():
                raise FileNotFoundError(
                    f"Baseline config.json not found at {self.base_config_path}"
                )
            self.config_path.write_text(self.base_config_path.read_text())

    def read_config(self) -> Dict[str, Any]:
        return json.loads(self.config_path.read_text())

    def write_config(self, cfg: Dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(cfg, indent=2))

    def run_train(self) -> Dict[str, Any]:
        subprocess.run(
            [sys.executable, str(self.train_script)],
            cwd=self.workspace,
            check=True,
        )
        return json.loads(self.results_path.read_text())
