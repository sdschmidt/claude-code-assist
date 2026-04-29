"""Tray-toggle settings persisted to ``<config_dir>/config.json``.

Lives in a top-level ``settings`` sub-object alongside the rest of the
``CompanionConfig`` keys. Owned exclusively by this module; the rest of
the config is owned by :func:`claude_code_assist.config.save_config`,
which preserves the ``settings`` block when it rewrites the file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "config.json"
_SETTINGS_KEY = "settings"


@dataclass
class CompanionSettings:
    gravity_enabled: bool = True
    walking_enabled: bool = True
    companion_scale: float = 1.0
    # Diagnostic logs — opt-in. Off by default so a normal install doesn't
    # accumulate prompt files; users with debugging or auditing needs can
    # flip them on from ``companion settings``.
    commentary_prompt_log: bool = False
    """Append every commentary call to ``<companion>/prompts.jsonl``."""
    art_prompt_log: bool = False
    """Save the Gemini sprite prompt next to ``sprite.png`` as ``prompt.txt``."""
    creation_prompt_log: bool = False
    """Save the profile-creation prompt to ``<companion>/creation_prompt.txt``."""


class SettingsStore:
    """Reads/writes the ``settings`` sub-object in ``config.json``."""

    def __init__(self, config_dir: Path) -> None:
        self._path = Path(config_dir) / _CONFIG_FILENAME

    def load(self) -> CompanionSettings:
        if not self._path.is_file():
            return CompanionSettings()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; using defaults", self._path, exc_info=True)
            return CompanionSettings()
        if not isinstance(data, dict):
            return CompanionSettings()
        block = data.get(_SETTINGS_KEY, {})
        if not isinstance(block, dict):
            return CompanionSettings()
        return CompanionSettings(
            gravity_enabled=bool(block.get("gravity_enabled", True)),
            walking_enabled=bool(block.get("walking_enabled", True)),
            companion_scale=float(block.get("companion_scale", 1.0)),
            commentary_prompt_log=bool(block.get("commentary_prompt_log", False)),
            art_prompt_log=bool(block.get("art_prompt_log", False)),
            creation_prompt_log=bool(block.get("creation_prompt_log", False)),
        )

    def save(self, settings: CompanionSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, object] = {}
        if self._path.is_file():
            try:
                text = self._path.read_text(encoding="utf-8")
                if text.strip():
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        existing = parsed
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read existing %s; rewriting from scratch", self._path, exc_info=True)
                existing = {}
        existing[_SETTINGS_KEY] = asdict(settings)
        self._path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
