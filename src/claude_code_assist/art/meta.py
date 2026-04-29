"""Per-art-set metadata (`meta.json` next to the frame PNGs)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

META_FILENAME = "meta.json"


class ArtMeta(BaseModel):
    """Provenance for a generated / prefilled art set."""

    datetime_of_creation: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = "unknown"
    prompt: str = ""

    model_config = {"arbitrary_types_allowed": True}


def load_meta(art_dir: Path) -> ArtMeta | None:
    """Load ``meta.json`` from ``art_dir``; return ``None`` if missing/invalid."""
    path = art_dir / META_FILENAME
    if not path.is_file():
        return None
    try:
        return ArtMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not load %s as ArtMeta", path, exc_info=True)
        return None


def write_meta(art_dir: Path, meta: ArtMeta) -> None:
    """Write ``meta.json`` into ``art_dir``."""
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = meta.model_dump(mode="json")
    (art_dir / META_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
