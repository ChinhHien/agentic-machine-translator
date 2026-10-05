"""Registry pattern mapping file extensions to loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger

from polyglot_lc.ingest.model import Document

Loader = Callable[[bytes], Document]

_REGISTRY: dict[str, Loader] = {}


def register(ext: str, loader: Loader) -> None:
    """Register *loader* for *ext* (e.g. ``register(".md", load_md)``)."""
    key = ext.lower().lstrip(".")
    _REGISTRY[key] = loader


def get_loader(ext: str) -> Loader:
    """Return the loader for *ext*, raising for unsupported types."""
    key = ext.lower().lstrip(".")
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported file extension {ext!r}. Supported: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def load_document(path: str | Path) -> Document:
    """Load *path* via the registered loader for its extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    loader = get_loader(p.suffix)
    data = p.read_bytes()
    doc = loader(data)
    doc.metadata["source"] = str(p)
    doc.metadata["filename"] = p.name
    logger.debug("Loaded {} ({} blocks)", p, len(doc.blocks))
    return doc


# Built-in loaders -------------------------------------------------------
def _bootstrap() -> None:
    from polyglot_lc.ingest import md as _md
    from polyglot_lc.ingest import pdf as _pdf
    from polyglot_lc.ingest import txt as _txt

    register("pdf", _pdf.load_pdf)
    register("txt", _txt.load_txt)
    register("text", _txt.load_txt)
    register("md", _md.load_md)
    register("markdown", _md.load_md)


_bootstrap()
