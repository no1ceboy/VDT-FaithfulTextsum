"""Locate complete local Transformers model folders.

Company-machine uploads commonly add one or more wrapper folders around a
model archive.  Hugging Face accepts a direct model folder, but it cannot
discover an arbitrary ``hf-cache/minicheck/config.json`` folder when that
folder is passed as ``cache_dir``.  These helpers deliberately inspect only
the requested tree and never contact the Hub.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

# Hub cache blob directories contain individual files, not model roots.  They
# are both expensive and unsafe to scan as candidate model directories.
_SKIP_DIRECTORY_NAMES = {
    ".git",
    ".locks",
    "__pycache__",
    "blobs",
    "refs",
}


def iter_candidate_dirs(root: str | Path, *, max_depth: int = 8) -> Iterator[Path]:
    """Yield directories below ``root`` without leaving the requested tree."""
    requested = Path(root).expanduser()
    if not requested.is_dir():
        return

    try:
        requested = requested.resolve()
    except OSError:
        requested = requested.absolute()

    queue: deque[tuple[Path, int]] = deque([(requested, 0)])
    visited: set[Path] = set()
    while queue:
        current, depth = queue.popleft()
        try:
            resolved = current.resolve()
        except OSError:
            resolved = current.absolute()
        if resolved in visited:
            continue
        visited.add(resolved)
        yield resolved
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (
                    child
                    for child in resolved.iterdir()
                    if child.is_dir() and child.name not in _SKIP_DIRECTORY_NAMES
                ),
                key=lambda child: child.name.lower(),
            )
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)


def has_complete_transformers_files(
    path: str | Path,
    *,
    tokenizer_filenames: tuple[str, ...],
) -> bool:
    """Return whether ``path`` has config, weights, and tokenizer assets."""
    folder = Path(path)
    return (
        (folder / "config.json").is_file()
        and any((folder / name).is_file() for name in MODEL_WEIGHT_FILENAMES)
        and any((folder / name).is_file() for name in tokenizer_filenames)
    )


def find_model_dir(
    root: str | Path,
    *,
    is_complete: Callable[[Path], bool],
    preferred_tokens: tuple[str, ...] = (),
    max_depth: int = 8,
) -> Path | None:
    """Find the best complete model folder below ``root``.

    A direct model folder wins immediately.  Otherwise candidates containing
    preferred tokens (for example ``minicheck`` or ``roberta-base``) and
    standard Hub snapshot folders are ranked first.  Only paths under the
    caller-provided root are considered.
    """
    requested = Path(root).expanduser()
    try:
        requested_resolved = requested.resolve()
    except OSError:
        requested_resolved = requested.absolute()
    if requested_resolved.is_dir() and is_complete(requested_resolved):
        return requested_resolved

    candidates = [
        candidate
        for candidate in iter_candidate_dirs(requested, max_depth=max_depth)
        if is_complete(candidate)
    ]
    if not candidates:
        return None

    def rank(candidate: Path) -> tuple[int, int, int]:
        try:
            relative_parts = candidate.relative_to(requested_resolved).parts
        except ValueError:
            relative_parts = candidate.parts
        haystack = str(candidate).lower()
        token_score = sum(10 for token in preferred_tokens if token.lower() in haystack)
        snapshot_score = 3 if candidate.parent.name.lower() == "snapshots" else 0
        # Prefer a shallower match when otherwise equivalent.
        depth_score = -len(relative_parts)
        return (token_score + snapshot_score, depth_score, int(candidate == requested_resolved))

    return max(candidates, key=rank).resolve()

