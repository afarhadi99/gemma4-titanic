"""Minimal .env loader (no third-party dependencies)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def load_env(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE lines into os.environ and return the pairs that were applied."""
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        if key in os.environ and not override:
            continue
        loaded[key] = os.environ[key] = _unquote(value.strip())
    return loaded


def missing(keys: Iterable[str]) -> list[str]:
    """Return the keys that are unset or empty in os.environ."""
    return [key for key in keys if not os.environ.get(key)]
