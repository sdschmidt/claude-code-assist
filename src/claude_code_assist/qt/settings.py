"""Tray + CLI settings persisted to ``<config_dir>/settings.json``.

Owns its own file (``settings.json``) — separate from the model-field
``config.json`` owned by :class:`CompanionConfig`. Earlier versions
stored these in a ``settings`` sub-object inside ``config.json``;
:meth:`SettingsStore.load` falls back to that block when ``settings.json``
doesn't exist yet so existing installs migrate transparently on first
save.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_SETTINGS_FILENAME = "settings.json"
_LEGACY_CONFIG_FILENAME = "config.json"
_LEGACY_SETTINGS_KEY = "settings"

# Claude Code settings sources the Agent SDK will load when our
# companion calls Claude. ``"default"`` means "pass nothing" — the SDK
# decides; the others map 1:1 to its ``setting_sources`` enum.
ClaudeSettingSource = Literal["default", "user", "project", "local"]
CLAUDE_SETTING_SOURCE_VALUES: tuple[ClaudeSettingSource, ...] = ("default", "user", "project", "local")


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
    # Which Claude Code settings file the Agent SDK should load when the
    # companion's provider is ``claude``. ``default`` passes ``None`` to
    # the SDK; the others pass ``[<source>]``.
    claude_setting_sources: ClaudeSettingSource = field(default="default")


def settings_to_sdk_arg(value: ClaudeSettingSource) -> list[str] | None:
    """Convert the user-facing setting to the SDK's ``setting_sources`` arg.

    ``"default"`` → ``None`` (SDK decides). Others → single-source list.
    """
    if value == "default":
        return None
    return [value]


class SettingsStore:
    """Reads/writes ``<config_dir>/settings.json``.

    Migrates from the legacy ``config.json`` ``settings`` block on first
    load when no ``settings.json`` exists yet — the legacy block stays
    behind in ``config.json`` until the next ``save_config`` call drops
    it.
    """

    def __init__(self, config_dir: Path) -> None:
        self._dir = Path(config_dir)
        self._path = self._dir / _SETTINGS_FILENAME
        self._legacy_path = self._dir / _LEGACY_CONFIG_FILENAME

    def load(self) -> CompanionSettings:
        block = self._read_block()
        if block is None:
            return CompanionSettings()
        raw_source = block.get("claude_setting_sources", "default")
        if raw_source not in CLAUDE_SETTING_SOURCE_VALUES:
            raw_source = "default"
        return CompanionSettings(
            gravity_enabled=bool(block.get("gravity_enabled", True)),
            walking_enabled=bool(block.get("walking_enabled", True)),
            companion_scale=float(block.get("companion_scale", 1.0)),
            commentary_prompt_log=bool(block.get("commentary_prompt_log", False)),
            art_prompt_log=bool(block.get("art_prompt_log", False)),
            creation_prompt_log=bool(block.get("creation_prompt_log", False)),
            claude_setting_sources=raw_source,  # type: ignore[arg-type]
        )

    def save(self, settings: CompanionSettings) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")

    def _read_block(self) -> dict[str, object] | None:
        """Read the settings dict from ``settings.json`` or the legacy block."""
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read %s; using defaults", self._path, exc_info=True)
                return None
            return data if isinstance(data, dict) else None

        # Legacy fallback: settings used to live as a sub-object of config.json.
        if self._legacy_path.is_file():
            try:
                data = json.loads(self._legacy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(data, dict):
                block = data.get(_LEGACY_SETTINGS_KEY)
                if isinstance(block, dict):
                    return block
        return None
