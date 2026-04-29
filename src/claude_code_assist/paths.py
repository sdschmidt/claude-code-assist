"""Default config-directory resolution + one-shot legacy-name migration.

Centralized so every CLI entry point reaches the same location without
each one re-implementing XDG resolution. Stdlib-only on purpose: this
module is imported during arg parsing, before pydantic / the LLM
provider stack get pulled in by ``config.py``.

The directory was renamed from ``claude-code-assist`` to
``claude-companion`` once the project rebranded around the desktop
companion. :func:`default_config_dir` performs an idempotent rename of
the legacy directory the first time it sees one — users keep their
roster, profile.json, art, and config.json without manual moves.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR_NAME = "claude-companion"
_LEGACY_DIR_NAME = "claude-code-assist"


def _xdg_base() -> Path:
    """Return ``$XDG_CONFIG_HOME`` if set, else ``~/.config``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def default_config_dir() -> Path:
    """Return the default config dir, migrating from the legacy name if needed.

    Migration policy: if ``<base>/claude-code-assist`` exists and
    ``<base>/claude-companion`` does not, atomically rename the former
    to the latter. If both exist (e.g. the user already created or
    migrated the new dir), leave the legacy directory alone and use
    the new one — touching it would risk clobbering newer data. If
    the rename fails (cross-device move on a weird mount, permission
    error), fall back to the legacy path so the app still launches.
    """
    base = _xdg_base()
    new = base / _DIR_NAME
    legacy = base / _LEGACY_DIR_NAME

    if legacy.is_dir() and not new.exists():
        try:
            legacy.rename(new)
            logger.info("Migrated config dir %s → %s", legacy, new)
        except OSError:
            logger.exception("Could not migrate %s → %s; falling back to legacy path", legacy, new)
            return legacy
    return new
