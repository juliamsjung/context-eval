from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Root directory for all traces
TRACES_ROOT = Path("traces")

def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with seconds precision and trailing Z."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _default_run_id(task_id: str) -> str:
    """
    Construct a default run_id based on task_id and current timestamp.
    Example: 'nomad_2025-12-03T01-23-45Z'
    """
    ts = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
    return f"{task_id}_{ts}Z"

@dataclass
class RunLogger:
    """
    Run-scoped logger that writes JSONL events for a single run.

    Public schema: every event is a JSON object with keys:
      - run_id: str
      - event_type: str  (e.g., 'run.start', 'op.train', 'step.summary')
      - step_idx: int | null
      - timestamp: ISO8601 string
      - task_id: str
      - dataset_id: str
      - agent_id: str
      - details: dict[str, Any]  (event-specific payload)
    """

    run_id: str
    task_id: str
    dataset_id: str
    agent_id: str
    path: Path

    def _write(self, event_type: str, step_idx: Optional[int], details: dict[str, Any]) -> None:
        """
        Low-level helper that writes a single JSONL event to the run file (append-only).
        """
        record = {
            "run_id": self.run_id,
            "event_type": event_type,
            "step_idx": step_idx,
            "timestamp": _now_iso(),
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "agent_id": self.agent_id,
            "details": details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ===== High-level convenience methods =====

    def log_run_start(
        self,
        *,
        config_hash: str,
        max_steps: int,
        seed: int,
        notes: str = "",
    ) -> None:
        """
        Log the start of a run. This should be called exactly once per run.
        """
        self._write(
            event_type="run.start",
            step_idx=None,
            details={
                "config_hash": config_hash,
                "max_steps": max_steps,
                "seed": seed,
                "notes": notes,
            },
        )

    def log_run_end(
        self,
        *,
        status: str,
        final_metric: float | None,
        best_step_idx: int | None,
        n_steps: int,
    ) -> None:
        """
        Log the end of a run. This should be called exactly once per run.
        """
        self._write(
            event_type="run.end",
            step_idx=best_step_idx,
            details={
                "status": status,  # e.g. 'success', 'error', 'timeout'
                "final_metric": final_metric,
                "best_step_idx": best_step_idx,
                "n_steps": n_steps,
            },
        )

    def log_op(self, event_type: str, step_idx: int, details: dict[str, Any]) -> None:
        """
        Log an operation-level event for a specific step/iteration.

        event_type should be a string like:
          - 'op.data_load'
          - 'op.config_proposal'
          - 'op.train'
          - 'op.eval'
          - 'op.submission'
        """
        if not event_type.startswith("op."):
            raise ValueError(f"Operation event_type should start with 'op.': {event_type}")
        self._write(event_type=event_type, step_idx=step_idx, details=details)

    def log_step_summary(self, step_idx: int, details: dict[str, Any]) -> None:
        """
        Log a summary for a given step. This is the per-iteration rollup and is expected
        to include:
          - metrics: { ... }
          - config: { ... }
          - decision: { ... }
          - context: { ... }
        but we do not enforce the inner structure here to keep this generic.
        """
        self._write(event_type="step.summary", step_idx=step_idx, details=details)

def start_run(
    *,
    task_id: str,
    dataset_id: str,
    agent_id: str,
    run_id: Optional[str] = None,
) -> RunLogger:
    """
    Factory to create a RunLogger for a single run.

    - task_id: logical task name (e.g. 'toy_tabular', 'nomad')
    - dataset_id: dataset name (often same as task_id, can differ for shared datasets)
    - agent_id: model/agent identifier (e.g. 'gpt-4.1', 'gpt-5.1', 'baseline_rf')
    - run_id: optional explicit run_id; otherwise auto-generated from task_id + timestamp

    The resulting JSONL file lives at:
        traces/{task_id}/{run_id}.jsonl
    """
    if run_id is None:
        run_id = _default_run_id(task_id)
    path = TRACES_ROOT / task_id / f"{run_id}.jsonl"
    return RunLogger(
        run_id=run_id,
        task_id=task_id,
        dataset_id=dataset_id,
        agent_id=agent_id,
        path=path,
    )
