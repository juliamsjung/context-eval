from pathlib import Path

# Project root (context-eval/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Logging root
LOGS_ROOT = PROJECT_ROOT / "logs"

# Subdirectories
TRACES_ROOT = LOGS_ROOT / "traces"
RUNS_ROOT = LOGS_ROOT / "runs"
