"""Repository-root anchoring and safe child-path resolution for the web UI."""

from __future__ import annotations

from pathlib import Path

#: The repository root, derived from this file's own location
#: (``<root>/webui/paths.py``) so it does not depend on the current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Characters that can never be part of a single file name.
_SEPARATORS = ("/", "\\")


def is_within(root: Path, candidate: Path) -> bool:
    """True when *candidate* resolves to *root* or to something inside it."""
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
    except OSError:  # pragma: no cover - exotic or unreadable paths
        return False
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def safe_child(root: Path, name: str) -> Path:
    """Join *name* onto *root*, refusing any name that could escape it.

    Names arrive from URLs, so the name is judged on its own characters first —
    a separator or a ``..`` component is never legitimate inside a single file
    name — and the joined path is then checked again after resolution, so a
    symlink cannot smuggle the caller outside *root* either.
    """
    if not name or any(mark in name for mark in _SEPARATORS):
        raise ValueError(f"unsafe name: {name!r}")
    if name in {".", ".."} or ".." in Path(name).parts:
        raise ValueError(f"unsafe name: {name!r}")
    candidate = root / name
    if not is_within(root, candidate):
        raise ValueError(f"path escapes {root}: {name!r}")
    return candidate
