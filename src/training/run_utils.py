"""Small shared helpers for local, reproducible training runs."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRL_TARGET_VERSION = "1.9.2"


def resolve_path(value: str | Path) -> Path:
    """Resolve relative paths from the repository root, not the caller's cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def write_json(path: Path, value: dict) -> None:
    """Atomically write a UTF-8 JSON manifest/config file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
