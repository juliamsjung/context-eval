"""Configuration and environment utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

_ENV_VARS: Dict[str, str] = {}
_ENV_LOADED = False


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from a .env file if present."""
    global _ENV_VARS, _ENV_LOADED

    if _ENV_LOADED:
        return _ENV_VARS

    path = env_path or DEFAULT_ENV_PATH
    if not path.exists():
        _ENV_LOADED = True
        return _ENV_VARS

    loaded_vars: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            loaded_vars[key] = value

    _ENV_VARS = loaded_vars
    _ENV_LOADED = True
    return _ENV_VARS


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a value from the loaded .env data or environment."""
    load_env_file()
    # First check loaded .env vars, then fall back to os.environ
    if key in _ENV_VARS:
        return _ENV_VARS[key]
    return os.environ.get(key, default)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(Path(path).read_text())
