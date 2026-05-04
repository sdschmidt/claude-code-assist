"""Profile persistence + roster-based config-dir layout.

Layout (current):

    <config_dir>/
        config.json              # provider settings + ``active_companion`` + tray toggles
        debug.log
        roster/
            <CompanionName>/
                profile.json
                art/
                    frame_{0..9}.png
                    icon.png
                    sprite.png
                    prompt.txt
                    meta.json
                art_archive/<ts>/   # previous art sets after `companion art` regen

The ``active_companion`` field of ``config.json`` names which roster
entry the runtime + ``companion art`` operate on. Switching companions
is a single config write — no file moves. Generation appends a numeric
suffix (``Pixie_2``, ``Pixie_3`` …) when a name collides.

Earlier layouts had ``<config>/companion/`` for the active and
``<config>/archive/<ts>_<name>/`` for archived entries (and before
that, profile + art at the top of the config dir). All are folded
into the current layout by :func:`migrate_legacy_layout` on every
launch — idempotent.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from claude_code_assist.io import load_json, save_json
from claude_code_assist.models.companion import CompanionProfile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

ROSTER_DIR_NAME = "roster"
ART_DIR_NAME = "art"
ART_ARCHIVE_DIR_NAME = "art_archive"
PROFILE_FILENAME = "profile.json"
LEGACY_PROFILE_FILENAME = "profile.yaml"
CONFIG_FILENAME = "config.json"

LEGACY_COMPANION_DIR_NAME = "companion"
LEGACY_PET_DIR_NAME = "pet"
LEGACY_ARCHIVE_DIR_NAME = "archive"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def roster_dir(config_dir: Path) -> Path:
    """Return ``<config_dir>/roster``."""
    return config_dir / ROSTER_DIR_NAME


def companion_path(config_dir: Path, slot: str) -> Path:
    """Return ``<config_dir>/roster/<slot>`` for a known roster slot name."""
    return roster_dir(config_dir) / slot


def companion_art_dir(config_dir: Path, slot: str | None = None) -> Path:
    """Return ``<config_dir>/roster/<slot>/art``.

    If ``slot`` is ``None`` the active slot is looked up; falls back to a
    placeholder ``_active`` directory if no companion is active yet so
    callers always get a non-``None`` ``Path`` to reason about.
    """
    if slot is None:
        active = get_active_slot(config_dir)
        if active is None:
            return roster_dir(config_dir) / "_active" / ART_DIR_NAME
        slot = active
    return companion_path(config_dir, slot) / ART_DIR_NAME


def companion_art_archive_dir(config_dir: Path, slot: str | None = None) -> Path:
    """Return ``<config_dir>/roster/<slot>/art_archive`` — past art sets per companion."""
    if slot is None:
        active = get_active_slot(config_dir)
        if active is None:
            return roster_dir(config_dir) / "_active" / ART_ARCHIVE_DIR_NAME
        slot = active
    return companion_path(config_dir, slot) / ART_ARCHIVE_DIR_NAME


def list_roster(config_dir: Path) -> list[Path]:
    """Return roster slots that contain a ``profile.json`` (alphabetical)."""
    root = roster_dir(config_dir)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir() and (entry / PROFILE_FILENAME).is_file():
            out.append(entry)
    return out


def find_companion_dir(config_dir: Path, query: str) -> Path | None:
    """Case-insensitive lookup of a roster slot by folder name."""
    target = query.strip().lower()
    if not target:
        return None
    for entry in list_roster(config_dir):
        if entry.name.lower() == target:
            return entry
    return None


def allocate_companion_slot(config_dir: Path, name: str) -> Path:
    """Return a *new*, unused roster directory path for ``name``.

    Sanitizes ``name`` into a filesystem-safe folder. Appends ``_2``,
    ``_3`` … on collision so two companions with the same display
    name don't fight over a slot.
    """
    safe = _SAFE_NAME_RE.sub("-", name).strip("-") or "companion"
    candidate = companion_path(config_dir, safe)
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = companion_path(config_dir, f"{safe}_{suffix}")
    return candidate


# ---------------------------------------------------------------------------
# Active-companion accessors
# ---------------------------------------------------------------------------


def _read_config_raw(config_dir: Path) -> dict[str, Any]:
    path = config_dir / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not parse %s; treating as empty", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _write_config_raw(config_dir: Path, data: dict[str, Any]) -> None:
    path = config_dir / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_active_slot(config_dir: Path) -> str | None:
    """Return the ``active_companion`` slot name, or ``None``."""
    value = _read_config_raw(config_dir).get("active_companion")
    return value if isinstance(value, str) and value else None


def set_active_slot(config_dir: Path, slot: str | None) -> None:
    """Persist ``active_companion`` in ``config.json``.

    Activating a slot also stamps ``profile.last_activated_at`` so the
    roster picker can sort by recency-of-use. Best-effort: a missing
    or unreadable profile silently skips the stamp.
    """
    data = _read_config_raw(config_dir)
    if slot is None:
        data.pop("active_companion", None)
    else:
        data["active_companion"] = slot
    _write_config_raw(config_dir, data)

    if slot is None:
        return
    profile_path = companion_path(config_dir, slot) / PROFILE_FILENAME
    profile = load_profile(profile_path)
    if profile is None:
        return
    profile.last_activated_at = datetime.now(UTC)
    save_profile(profile, profile_path)


def get_active_companion_dir(config_dir: Path) -> Path | None:
    """Return the active companion's roster directory, or ``None``."""
    slot = get_active_slot(config_dir)
    if slot is None:
        return None
    candidate = companion_path(config_dir, slot)
    if not candidate.is_dir() or not (candidate / PROFILE_FILENAME).is_file():
        return None
    return candidate


# ---------------------------------------------------------------------------
# Profile load / save
# ---------------------------------------------------------------------------


def save_profile(companion: CompanionProfile, path: Path) -> None:
    """Save a companion profile to JSON, creating parent dirs as needed."""
    save_json(companion, path)
    logger.info("Saved profile to %s", path)


def load_profile(path: Path) -> CompanionProfile | None:
    """Load a companion profile from JSON or return ``None`` if missing/invalid."""
    return load_json(path, CompanionProfile)


def get_profile_path(config_dir: Path, *, slot: str | None = None) -> Path:
    """Path to a companion's ``profile.json``."""
    if slot is None:
        active = get_active_slot(config_dir)
        if active is None:
            return roster_dir(config_dir) / "_active" / PROFILE_FILENAME
        slot = active
    return companion_path(config_dir, slot) / PROFILE_FILENAME


# ---------------------------------------------------------------------------
# Per-companion art archival
# ---------------------------------------------------------------------------


def archive_current_art(config_dir: Path, *, slot: str | None = None) -> Path | None:
    """Move the active companion's ``art/`` to ``art_archive/<ts>/``."""
    if slot is None:
        slot = get_active_slot(config_dir)
        if slot is None:
            return None
    src = companion_art_dir(config_dir, slot)
    if not src.is_dir() or not any(src.iterdir()):
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archive_root = companion_art_archive_dir(config_dir, slot)
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / timestamp
    suffix = 0
    while target.exists():
        suffix += 1
        target = archive_root / f"{timestamp}_{suffix}"
    shutil.move(str(src), str(target))
    logger.info("Archived current art set to %s", target)
    return target


# ---------------------------------------------------------------------------
# Legacy layout migration
# ---------------------------------------------------------------------------


def migrate_legacy_layout(config_dir: Path) -> bool:
    """Bring legacy config layouts into the current ``roster/`` layout."""
    moved = False
    if not config_dir.exists():
        return moved

    target_companion = config_dir / LEGACY_COMPANION_DIR_NAME

    legacy_pet = config_dir / LEGACY_PET_DIR_NAME
    if legacy_pet.is_dir() and not target_companion.exists():
        shutil.move(str(legacy_pet), str(target_companion))
        logger.info("Renamed %s → %s", legacy_pet, target_companion)
        moved = True

    legacy_profile_top = config_dir / LEGACY_PROFILE_FILENAME
    legacy_profile_in_companion = target_companion / LEGACY_PROFILE_FILENAME
    new_profile_in_companion = target_companion / PROFILE_FILENAME
    if (
        legacy_profile_top.is_file()
        and not legacy_profile_in_companion.exists()
        and not new_profile_in_companion.exists()
    ):
        target_companion.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_profile_top), str(legacy_profile_in_companion))
        logger.info("Migrated %s → %s", LEGACY_PROFILE_FILENAME, legacy_profile_in_companion)
        moved = True

    target_art = target_companion / ART_DIR_NAME
    legacy_art = config_dir / "art"
    if legacy_art.is_dir() and not target_art.exists():
        target_companion.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_art), str(target_art))
        logger.info("Migrated art/ → %s", target_art)
        moved = True

    for stray in config_dir.glob("*_icon_*.png"):
        if stray.is_file():
            target_art.mkdir(parents=True, exist_ok=True)
            dest = target_art / stray.name
            if not dest.exists():
                shutil.move(str(stray), str(dest))
                moved = True

    if _migrate_profile_to_json(target_companion):
        moved = True

    if _strip_art_name_prefix(target_art, target_companion / PROFILE_FILENAME):
        moved = True

    if _migrate_to_roster_layout(config_dir):
        moved = True

    if _migrate_config_to_json(config_dir):
        moved = True

    if _migrate_icon_filename(config_dir):
        moved = True

    return moved


def _migrate_icon_filename(config_dir: Path) -> bool:
    """Rename ``icon_64.png`` → ``icon.png`` across all roster art and archive dirs."""
    roster_root = roster_dir(config_dir)
    if not roster_root.is_dir():
        return False
    renamed = False
    for slot_dir in roster_root.iterdir():
        if not slot_dir.is_dir():
            continue
        candidates = [slot_dir / ART_DIR_NAME]
        archive_root = slot_dir / ART_ARCHIVE_DIR_NAME
        if archive_root.is_dir():
            candidates.extend(p for p in archive_root.iterdir() if p.is_dir())
        for art_path in candidates:
            legacy = art_path / "icon_64.png"
            if not legacy.is_file():
                continue
            target = art_path / "icon.png"
            if target.exists():
                continue
            try:
                legacy.rename(target)
            except OSError:
                logger.exception("Could not rename %s → %s", legacy, target)
                continue
            renamed = True
    return renamed


def _migrate_to_roster_layout(config_dir: Path) -> bool:
    """Move legacy ``companion/`` + ``archive/<ts>_<name>/`` into ``roster/<slot>/``."""
    legacy_active = config_dir / LEGACY_COMPANION_DIR_NAME
    legacy_archive_root = config_dir / LEGACY_ARCHIVE_DIR_NAME

    has_legacy_active = legacy_active.is_dir() and (legacy_active / PROFILE_FILENAME).is_file()
    legacy_archive_entries: list[Path] = []
    if legacy_archive_root.is_dir():
        for entry in legacy_archive_root.iterdir():
            if entry.is_dir() and (entry / PROFILE_FILENAME).is_file():
                legacy_archive_entries.append(entry)

    if not has_legacy_active and not legacy_archive_entries:
        return False

    moved = False
    new_active_slot: str | None = None

    if has_legacy_active:
        target = _next_roster_slot_for_dir(config_dir, legacy_active)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_active), str(target))
        new_active_slot = target.name
        logger.info("Migrated active companion → %s", target)
        moved = True

    for legacy_entry in sorted(legacy_archive_entries, key=lambda p: p.name):
        target = _next_roster_slot_for_dir(config_dir, legacy_entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_entry), str(target))
        logger.info("Migrated archived companion %s → %s", legacy_entry.name, target)
        moved = True

    if legacy_archive_root.is_dir():
        for stray in list(legacy_archive_root.iterdir()):
            if stray.name == ".DS_Store":
                stray.unlink(missing_ok=True)
        try:
            legacy_archive_root.rmdir()
        except OSError:
            pass

    if new_active_slot is not None:
        set_active_slot(config_dir, new_active_slot)

    return moved


def _next_roster_slot_for_dir(config_dir: Path, source: Path) -> Path:
    profile_path = source / PROFILE_FILENAME
    profile = load_profile(profile_path)
    name = profile.name if profile is not None else source.name
    return allocate_companion_slot(config_dir, name)


def _migrate_profile_to_json(companion_subdir: Path) -> bool:
    """Convert ``<dir>/profile.yaml`` → ``<dir>/profile.json`` once."""
    yaml_path = companion_subdir / LEGACY_PROFILE_FILENAME
    json_path = companion_subdir / PROFILE_FILENAME
    if not yaml_path.is_file() or json_path.exists():
        return False

    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, ImportError) as exc:
        logger.warning("Could not read legacy %s: %s", yaml_path, exc)
        return False
    except Exception:  # noqa: BLE001
        logger.warning("Could not parse legacy %s; leaving in place", yaml_path, exc_info=True)
        return False

    if not isinstance(data, dict):
        return False

    data.pop("project_path", None)

    try:
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        return False

    try:
        yaml_path.unlink()
    except OSError:
        pass

    logger.info("Migrated %s → %s", yaml_path, json_path)
    return True


def _strip_art_name_prefix(art_dir: Path, profile_path: Path) -> bool:
    """Drop ``{name}_`` from filenames in ``art_dir``."""
    if not art_dir.is_dir() or not profile_path.is_file():
        return False
    profile = load_profile(profile_path)
    if profile is None or not profile.name:
        return False
    prefix = f"{profile.name}_"
    renamed = False
    for src in list(art_dir.iterdir()):
        if not src.is_file() or not src.name.startswith(prefix):
            continue
        dest = src.with_name(src.name[len(prefix) :])
        if dest.exists():
            continue
        src.rename(dest)
        renamed = True
    return renamed


_LEGACY_SETTINGS_FILENAMES: tuple[str, ...] = ("companion_settings.json", "deskpet_settings.json")


def _migrate_config_to_json(config_dir: Path) -> bool:
    """Fold legacy ``config.yaml`` + ``*_settings.json`` into ``config.json``."""
    config_json = config_dir / CONFIG_FILENAME
    config_yaml = config_dir / "config.yaml"
    legacy_settings_paths = [config_dir / name for name in _LEGACY_SETTINGS_FILENAMES]
    has_legacy_settings = any(p.is_file() for p in legacy_settings_paths)

    if not config_yaml.is_file() and not has_legacy_settings:
        return False

    raw: dict[str, Any] = {}
    if config_json.is_file():
        try:
            text = config_json.read_text(encoding="utf-8")
            if text.strip():
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    raw = loaded
        except (OSError, json.JSONDecodeError):
            raw = {}

    if config_yaml.is_file() and not config_json.is_file():
        try:
            import yaml

            yaml_data = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(yaml_data, dict):
                raw = {**yaml_data, **raw}
        except (OSError, ImportError):
            pass
        except Exception:  # noqa: BLE001
            pass

    for legacy_path in legacy_settings_paths:
        if not legacy_path.is_file():
            continue
        try:
            settings_text = legacy_path.read_text(encoding="utf-8")
            settings_data = json.loads(settings_text) if settings_text.strip() else {}
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(settings_data, dict):
            continue

        if "petScale" in settings_data and "companionScale" not in settings_data:
            settings_data["companionScale"] = settings_data.pop("petScale")
        else:
            settings_data.pop("petScale", None)

        existing_settings = raw.get("settings")
        if not isinstance(existing_settings, dict):
            existing_settings = {}
        raw["settings"] = {**settings_data, **existing_settings}

    try:
        config_json.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    except OSError:
        return False

    for legacy in (config_yaml, *legacy_settings_paths):
        if legacy.is_file():
            try:
                legacy.unlink()
            except OSError:
                pass

    return True


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_active_profile(config_dir: Path) -> tuple[Path, CompanionProfile | None]:
    active_dir = get_active_companion_dir(config_dir)
    if active_dir is None:
        return get_profile_path(config_dir), None
    profile_path = active_dir / PROFILE_FILENAME
    return profile_path, load_profile(profile_path)


def iter_roster_profiles(config_dir: Path) -> Iterable[tuple[Path, CompanionProfile | None]]:
    for slot_dir in list_roster(config_dir):
        yield slot_dir, load_profile(slot_dir / PROFILE_FILENAME)
