"""``companion art`` — generate sprite frames or pick a premade set.

Top-level menu:

* **Generate new** — call Gemini with a prompt built from the active
  companion's profile.

  * **Automatic** — use ``profile.body_plan`` / ``walk_description`` /
    ``fall_description`` / ``landing_description`` exactly as they are.
  * **Adapt** — open four ``questionary.text`` prompts pre-filled with
    those fields so the user can edit (or just hit Enter to accept).

* **Premade** — ask the configured profile LLM to match the active
  companion to one of the bundled ``assets/placeholder_frames/<slug>/``
  sets (shape > limb count > material > color), then copy that slug's
  frames + icon + meta.json into the companion's art dir. No image
  generation, but still one LLM call for the match.

Before either path writes anything, the previous ``companion/art/``
(if any) is moved to ``companion/art_archive/<ts>/`` via
:func:`profile.storage.archive_current_art`. That gives us a free
roll-back if the user doesn't like the new run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from rich.console import Console

from claude_code_assist.art.meta import ArtMeta, load_meta
from claude_code_assist.art.prompts import LocomotionOverrides
from claude_code_assist.cli._picker import PICKER_STYLE, bind_shortcuts, menu_title
from claude_code_assist.profile.storage import (
    PROFILE_FILENAME,
    archive_current_art,
    companion_art_archive_dir,
    companion_art_dir,
    get_active_companion_dir,
    load_profile,
    migrate_legacy_layout,
)

if TYPE_CHECKING:
    from claude_code_assist.art import PremadeOption
    from claude_code_assist.config import CompanionConfig
    from claude_code_assist.models.companion import CompanionProfile

logger = logging.getLogger(__name__)
console = Console()

# (value, shortcut, label, description).
_TOP_CHOICES_NO_ARCHIVE: tuple[tuple[str, str, str, str], ...] = (
    ("generate", "n", "generate new", "call Gemini with prompts from your profile"),
    ("premade", "p", "premade", "match a bundled placeholder set to your companion via LLM"),
)
_RECROP_CHOICE: tuple[str, str, str, str] = (
    "recrop",
    "e",
    "recrop",
    "re-extract frames from the saved sprite with custom options",
)
_RESTORE_CHOICE: tuple[str, str, str, str] = (
    "restore",
    "r",
    "restore",
    "switch back to a previously archived art set",
)
_QUIT_CHOICE: tuple[str, str, str, str] = ("quit", "q", "quit", "leave art untouched")
_GEN_MODE_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    ("auto", "a", "automatic", "use the profile's locomotion descriptors as-is"),
    ("adapt", "d", "adapt", "review and edit each prompt before generating"),
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="companion art",
        description="Generate sprite frames for the active companion (Gemini), or prefill placeholders.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory override. Defaults to $XDG_CONFIG_HOME/claude-companion.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _resolve_config_dir(override: Path | None) -> Path:
    if override is not None:
        return override
    from claude_code_assist.paths import default_config_dir  # noqa: PLC0415

    return default_config_dir()


# ---------------------------------------------------------------------------
# Pickers
# ---------------------------------------------------------------------------


def _pick_top_choice(*, allow_recrop: bool, allow_restore: bool) -> str | None:
    """Top-level menu. ``Recrop`` and ``Restore`` only appear when applicable."""
    rows: list[tuple[str, str, str, str]] = list(_TOP_CHOICES_NO_ARCHIVE)
    if allow_recrop:
        rows.append(_RECROP_CHOICE)
    if allow_restore:
        rows.append(_RESTORE_CHOICE)
    rows.append(_QUIT_CHOICE)
    choices = [
        questionary.Choice(title=menu_title(label, description, shortcut=key), value=value)
        for value, key, label, description in rows
    ]
    shortcut_map = {key: value for value, key, _, _ in rows}
    try:
        question = questionary.select(
            "How do you want to generate art?",
            choices=choices,
            style=PICKER_STYLE,
        )
        bind_shortcuts(question, shortcut_map)
        return question.ask()
    except (KeyboardInterrupt, EOFError):
        return None


def _pick_gen_mode() -> str | None:
    choices = [
        questionary.Choice(title=menu_title(label, description, shortcut=key), value=value)
        for value, key, label, description in _GEN_MODE_CHOICES
    ]
    shortcut_map = {key: value for value, key, _, _ in _GEN_MODE_CHOICES}
    try:
        question = questionary.select(
            "Use profile prompts as-is, or edit them first?",
            choices=choices,
            style=PICKER_STYLE,
        )
        bind_shortcuts(question, shortcut_map)
        return question.ask()
    except (KeyboardInterrupt, EOFError):
        return None


# ---------------------------------------------------------------------------
# Adapt flow — pre-fill the four locomotion fields and let the user edit
# ---------------------------------------------------------------------------


def _collect_overrides(companion: CompanionProfile) -> LocomotionOverrides | None:
    fields = (
        ("body_plan", "body plan (anatomy, limbs, body shape)", companion.body_plan),
        ("walk", "walk description (two-stride cycle, facing right)", companion.walk_description),
        ("fall", "fall description (mid-air behavior)", companion.fall_description),
        ("landing", "landing description (touchdown reaction)", companion.landing_description),
    )
    edited: dict[str, str] = {}
    for key, label, current in fields:
        # Single-line input: questionary's ``multiline=True`` requires
        # alt-Enter / Esc-Enter to submit, which is confusing when the
        # entire line of "press Enter to accept" instruction is right
        # there. The locomotion descriptors are short enough to edit on
        # one line; users who want a real newline can paste one in.
        try:
            value = questionary.text(
                f"{label}:",
                default=current or "",
                instruction="(Enter to accept, edit otherwise)",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None
        if value is None:
            return None
        edited[key] = value.strip()
    return LocomotionOverrides(
        body_plan=edited["body_plan"] or None,
        walk_description=edited["walk"] or None,
        fall_description=edited["fall"] or None,
        landing_description=edited["landing"] or None,
    )


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def _ensure_gemini_api_key(config_dir: Path) -> str | None:
    """Resolve GEMINI_API_KEY from env or .env, prompting the user if absent.

    A typed key is also written to ``<config>/.env`` so we don't ask
    again on the next run. Returns the key, or ``None`` if the user
    declined to provide one.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Try <config>/.env (loaded lazily — we don't want to reach for
    # python-dotenv unless we actually need to).
    env_path = config_dir / ".env"
    if env_path.is_file():
        try:
            from dotenv import dotenv_values  # noqa: PLC0415

            values = dotenv_values(env_path)
            key = (values.get("GEMINI_API_KEY") or "").strip()
            if key:
                os.environ["GEMINI_API_KEY"] = key
                return key
        except ImportError:
            pass

    console.print(f"[yellow]GEMINI_API_KEY not found in your environment or {env_path}.[/yellow]")
    try:
        typed = questionary.password(
            "Paste your Gemini API key (or leave blank to abort):",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if not typed:
        return None
    typed = typed.strip()
    if not typed:
        return None

    # Persist for next run.
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
        if "GEMINI_API_KEY" not in existing:
            sep = "" if not existing or existing.endswith("\n") else "\n"
            env_path.write_text(f"{existing}{sep}GEMINI_API_KEY={typed}\n", encoding="utf-8")
            console.print(f"[dim]Saved key to {env_path}.[/dim]")
    except OSError:
        logger.warning("Could not persist GEMINI_API_KEY to %s", env_path, exc_info=True)
    os.environ["GEMINI_API_KEY"] = typed
    return typed


# ---------------------------------------------------------------------------
# Generation paths
# ---------------------------------------------------------------------------


def _archive_existing(config_dir: Path, slot: str) -> None:
    archived = archive_current_art(config_dir, slot=slot)
    if archived is not None:
        console.print(f"[dim]Archived previous art set to {archived.name}.[/dim]")


# ---------------------------------------------------------------------------
# Pending-art atomic swap
# ---------------------------------------------------------------------------
#
# Generation / prefill write to a sibling ``art.pending/`` directory and
# only swap into ``art/`` once the new set is fully on disk. That way
# Ctrl-C, an API failure, or any other crash leaves the existing
# ``art/`` and ``art_archive/`` exactly as they were — the old bug was
# archiving up front and ending up with no usable art when the user
# cancelled mid-generation.

_PENDING_ART_SUFFIX = ".pending"


def _pending_art_dir(art_dir: Path) -> Path:
    """Return the sibling staging directory used for in-progress art."""
    return art_dir.parent / f"{art_dir.name}{_PENDING_ART_SUFFIX}"


def _discard_pending_art(pending_dir: Path) -> None:
    """Best-effort cleanup of an in-progress / cancelled art directory."""
    if pending_dir.exists():
        shutil.rmtree(pending_dir, ignore_errors=True)


def _commit_pending_art(pending_dir: Path, art_dir: Path, config_dir: Path, slot: str) -> None:
    """Atomically replace ``art_dir`` with ``pending_dir``.

    Archives the existing ``art_dir`` (when non-empty) into the slot's
    ``art_archive/`` first, then renames ``pending_dir`` into place.
    The empty-directory branch handles the freshly-created companion
    case where ``art/`` exists but has no files — ``rename`` would
    otherwise refuse to overwrite even an empty target.
    """
    archived = archive_current_art(config_dir, slot=slot)
    if archived is not None:
        console.print(f"[dim]Archived previous art set to {archived.name}.[/dim]")
    elif art_dir.exists():
        shutil.rmtree(art_dir)
    pending_dir.rename(art_dir)


_DEPENDENCY_HINT = (
    "Run [bold]uv tool install . --reinstall[/bold] (or "
    "[bold]uv tool upgrade claude-code-assist[/bold]) to refresh the "
    "tool environment with the new dependencies."
)


def _print_active_banner(companion: CompanionProfile) -> None:
    """One-liner identical in shape to a ``companion roster`` entry.

    ``Name ★★★★★ · creature_type · Lv. N`` — name + stars colored by
    rarity (via the canonical ``Rarity.color`` hex; Rich accepts
    ``#rrggbb`` directly), the rest dimmed.
    """
    from claude_code_assist.models.role import ROLE_CATALOG  # noqa: PLC0415

    color = companion.rarity.color
    role_block = ""
    if companion.role is not None:
        defn = ROLE_CATALOG.get(companion.role)
        role_color = defn.color if defn else "#888"
        role_block = f"  [dim]·[/dim]  [{role_color}]{companion.role.value}[/{role_color}]"
    console.print(
        f"[bold {color}]{companion.name}[/bold {color}]"
        f"  [{color}]{companion.rarity.stars}[/{color}]"
        f"  [dim]·  Lv. {companion.level}  ·  {companion.creature_type}[/dim]"
        f"{role_block}"
    )


def _has_complete_art(art_dir: Path) -> bool:
    """True only when all 10 ``frame_{N}.png`` files exist."""
    if not art_dir.is_dir():
        return False
    return all((art_dir / f"frame_{i}.png").is_file() for i in range(10))


def _has_sprite_source(config_dir: Path, slot: str) -> bool:
    """True when there's a ``sprite.png`` we could re-crop from.

    Looks at the active art dir first, then any archive subfolder. The
    Recrop entry is hidden when both come up empty — there's nothing to
    re-extract.
    """
    if (companion_art_dir(config_dir, slot) / "sprite.png").is_file():
        return True
    return any((archive_path / "sprite.png").is_file() for archive_path, _ in _list_art_archives(config_dir, slot))


# Archive subdirs are named ``YYYYMMDD-HHMMSS`` (or ``…_<n>`` on collisions).
_ART_ARCHIVE_DIR_RE = re.compile(r"^(\d{8}-\d{6})(?:_\d+)?$")


# ---------------------------------------------------------------------------
# Restore — pick an archived art set and swap it in
# ---------------------------------------------------------------------------


def _restore_choice_title(folder_name: str, meta: ArtMeta | None) -> list[tuple[str, str]]:
    """Build the picker label for an archive entry: name + suffix dimmed.

    With a meta record we render the human timestamp + model. Without
    one (legacy folder), the suffix falls back to the raw folder name.
    """
    if meta is None:
        return [
            ("", folder_name),
            ("fg:ansibrightblack", "  (no meta.json)"),
        ]
    when = meta.datetime_of_creation.astimezone().strftime("%Y-%m-%d %H:%M")
    return [
        ("", when),
        ("fg:ansibrightblack", f"  ({meta.model})"),
    ]


def _list_art_archives(config_dir: Path, slot: str) -> list[tuple[Path, ArtMeta | None]]:
    """Scan a slot's ``art_archive/`` newest-first, attaching meta when present."""
    root = companion_art_archive_dir(config_dir, slot)
    if not root.is_dir():
        return []
    entries: list[tuple[Path, ArtMeta | None]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not _ART_ARCHIVE_DIR_RE.match(entry.name):
            continue
        entries.append((entry, load_meta(entry)))
    # Newest first — by meta timestamp when available, else by folder name
    # (which is itself a sortable timestamp).
    entries.sort(
        key=lambda pair: (pair[1].datetime_of_creation if pair[1] is not None else None, pair[0].name),
        reverse=True,
    )
    return entries


def _run_restore(config_dir: Path, slot: str) -> int:
    archives = _list_art_archives(config_dir, slot)
    if not archives:
        console.print("[yellow]No archived art sets to restore.[/yellow]")
        return 0

    choices = [
        questionary.Choice(title=_restore_choice_title(path.name, meta), value=str(path)) for path, meta in archives
    ]
    try:
        selected = questionary.select("Choose an art set to restore:", choices=choices).ask()
    except (KeyboardInterrupt, EOFError):
        selected = None
    if selected is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0

    chosen = Path(selected)
    art_dir = companion_art_dir(config_dir, slot)
    # Archive the current art set first so nothing is lost.
    # ``archive_current_art`` no-ops if there's nothing in art/.
    _archive_existing(config_dir, slot)
    shutil.move(str(chosen), str(art_dir))
    chosen_meta = load_meta(art_dir)
    if chosen_meta is not None:
        when = chosen_meta.datetime_of_creation.astimezone().strftime("%Y-%m-%d %H:%M")
        console.print(f"[green]Restored art from {when} ({chosen_meta.model}).[/green]")
    else:
        console.print(f"[green]Restored art from {chosen.name}.[/green]")
    console.print("Run [bold]companion[/bold] to start the companion.")
    return 0


def _run_generate(
    companion: CompanionProfile,
    config_dir: Path,
    slot: str,
    *,
    overrides: LocomotionOverrides | None,
    api_key: str,
) -> int:
    try:
        from claude_code_assist.art import generate_frames
    except ImportError as exc:
        console.print(f"[red]Art generation dependencies are missing:[/red] {exc.name or exc}\n{_DEPENDENCY_HINT}")
        return 1
    art_dir = companion_art_dir(config_dir, slot)
    pending_dir = _pending_art_dir(art_dir)
    # Wipe a stale staging dir from a previous crashed / killed run so
    # we don't merge half-baked frames into a fresh attempt.
    _discard_pending_art(pending_dir)

    from claude_code_assist.qt.settings import SettingsStore  # noqa: PLC0415

    settings = SettingsStore(config_dir).load()

    try:
        with console.status("[cyan]Generating sprite frames with Gemini…[/cyan]", spinner="dots"):
            frames = generate_frames(
                companion,
                pending_dir,
                overrides=overrides,
                api_key=api_key,
                write_prompt_log=settings.art_prompt_log,
            )
    except KeyboardInterrupt:
        _discard_pending_art(pending_dir)
        console.print("[yellow]Generation cancelled. Existing art is unchanged.[/yellow]")
        return 130
    except (RuntimeError, ValueError) as exc:
        _discard_pending_art(pending_dir)
        console.print(f"[red]Generation failed:[/red] {exc}")
        return 1
    except Exception:
        _discard_pending_art(pending_dir)
        raise

    _commit_pending_art(pending_dir, art_dir, config_dir, slot)
    console.print(f"[green]Saved {len(frames)} frames to {art_dir}.[/green]")
    console.print("Run [bold]companion[/bold] to start the companion.")
    return 0


def _ask_yes_no(message: str, *, default: bool) -> bool | None:
    """Wrapper around ``questionary.confirm`` that returns ``None`` on cancel."""
    try:
        answer = questionary.confirm(message, default=default).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    return answer


def _ask_int(message: str, *, default: int, minimum: int = 0, maximum: int = 32) -> int | None:
    """Prompt for a non-negative int. Returns ``None`` on cancel."""

    def _validate(raw: str) -> bool | str:
        try:
            value = int(raw)
        except ValueError:
            return "Enter a whole number."
        if value < minimum or value > maximum:
            return f"Must be between {minimum} and {maximum}."
        return True

    try:
        answer = questionary.text(message, default=str(default), validate=_validate).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if answer is None:
        return None
    return int(answer)


def _run_recrop(config_dir: Path, slot: str) -> int:
    """Re-extract frames from the saved ``sprite.png`` with custom toggles.

    Operates in place — the existing ``sprite.png`` and ``meta.json``
    are untouched, only ``frame_{N}.png`` are overwritten. No archival
    happens here: the sheet is the source of truth, and recrops are
    cheap to redo with different toggles. If the user wants the
    pre-recrop frames back, ``companion art → Restore`` still works
    because earlier generations were already archived.
    """
    art_dir = companion_art_dir(config_dir, slot)
    sprite_path = art_dir / "sprite.png"
    if not sprite_path.is_file():
        # Fall back to the most recent archive that has a sprite —
        # useful right after a Restore, where the active dir might not
        # yet have the source.
        for archive_path, _ in _list_art_archives(config_dir, slot):
            archive_sprite = archive_path / "sprite.png"
            if archive_sprite.is_file():
                sprite_path = archive_sprite
                console.print(
                    f"[dim]Using sprite.png from archive {archive_path.name} (no current art/sprite.png).[/dim]"
                )
                break
        else:
            console.print(
                f"[red]No sprite.png found at {art_dir / 'sprite.png'} or in any archive.[/red] "
                "Run [bold]companion art → Generate new[/bold] first."
            )
            return 1

    remove_grid = _ask_yes_no("Remove grid lines (paint over dark bands before splitting)?", default=True)
    if remove_grid is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0
    smart_split = _ask_yes_no(
        "Smart-find frames (detect cell boundaries from magenta corridors)?",
        default=True,
    )
    if smart_split is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0
    contiguous_chroma = _ask_yes_no(
        "Background flood-fill: contiguous-only? (yes = preserve interior chroma; no = also clear enclosed pockets)",
        default=False,
    )
    if contiguous_chroma is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0
    feather_px = _ask_int(
        "Feather edge (pixels of inward alpha ramp to kill magenta halo; 0 disables)",
        default=2,
    )
    if feather_px is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0

    # Lazy import so a missing pillow/numpy gives the friendly hint.
    try:
        from PIL import Image  # noqa: PLC0415

        from claude_code_assist.art import split_and_clean
    except ImportError as exc:
        console.print(f"[red]Recrop dependencies missing:[/red] {exc.name or exc}\n{_DEPENDENCY_HINT}")
        return 1

    art_dir.mkdir(parents=True, exist_ok=True)
    sprite = Image.open(sprite_path)
    # If we sourced the sprite from an archive (no active sprite),
    # copy it into the active dir so future recrops don't need the
    # archive fallback.
    target_sprite = art_dir / "sprite.png"
    if sprite_path.resolve() != target_sprite.resolve():
        target_sprite.write_bytes(sprite_path.read_bytes())

    try:
        frames = split_and_clean(
            sprite,
            art_dir,
            remove_grid=remove_grid,
            smart_split=smart_split,
            contiguous_chroma=contiguous_chroma,
            feather_px=feather_px,
        )
    except RuntimeError as exc:
        console.print(f"[red]Recrop failed:[/red] {exc}")
        return 1

    console.print(
        f"[green]Re-cropped {len(frames)} frames into {art_dir}.[/green]  "
        f"[dim](remove_grid={remove_grid}, smart={smart_split}, "
        f"contiguous={contiguous_chroma}, feather={feather_px}px)[/dim]"
    )
    console.print("Run [bold]companion[/bold] to start the companion.")
    return 0


def _load_companion_config(config_dir: Path) -> CompanionConfig:
    """Load ``config.json`` for LLM provider settings, defaulting to a fresh config."""
    from claude_code_assist.config import CompanionConfig as _Config  # noqa: PLC0415
    from claude_code_assist.config import load_config  # noqa: PLC0415

    cfg_path = config_dir / "config.json"
    if cfg_path.is_file():
        return load_config(cfg_path).model_copy(update={"config_dir": config_dir})
    return _Config(config_dir=config_dir)


def _pick_premade_manually(
    options: tuple[PremadeOption, ...], default_slug: str
) -> PremadeOption | None:
    """Manual override picker — used when the user rejects the LLM match."""
    choices = [
        questionary.Choice(
            title=f"{opt.slug}  —  {opt.description[:80]}",
            value=opt.slug,
        )
        for opt in options
    ]
    try:
        slug = questionary.select(
            "Pick a premade set:",
            choices=choices,
            default=default_slug,
            style=PICKER_STYLE,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if slug is None:
        return None
    return next((o for o in options if o.slug == slug), None)


def _run_premade(config_dir: Path, slot: str, companion: CompanionProfile) -> int:
    try:
        from claude_code_assist.art import (  # noqa: PLC0415
            copy_premade,
            list_premade_options,
            match_premade,
        )
    except ImportError as exc:
        console.print(
            f"[red]Premade copy needs a dependency that's missing:[/red] {exc.name or exc}\n{_DEPENDENCY_HINT}"
        )
        return 1

    options = list_premade_options()
    if not options:
        console.print("[red]No premade art sets are bundled.[/red]")
        return 1

    cfg = _load_companion_config(config_dir)

    try:
        with console.status("[cyan]Matching profile to premade art…[/cyan]", spinner="dots"):
            chosen, reason = match_premade(companion, options, cfg)
    except KeyboardInterrupt:
        console.print("[yellow]Match cancelled. Existing art is unchanged.[/yellow]")
        return 130
    except RuntimeError as exc:
        console.print(f"[yellow]Matcher failed:[/yellow] {exc}")
        chosen = options[0]
        reason = f"(LLM match failed; falling back to {options[0].slug})"

    console.print(f"[bold green]Match:[/bold green] {chosen.slug}")
    if reason:
        console.print(f"[dim]{reason}[/dim]")

    use_match = _ask_yes_no("Use this premade set?", default=True)
    if use_match is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0
    if not use_match:
        manual = _pick_premade_manually(options, chosen.slug)
        if manual is None:
            console.print("[yellow]Cancelled.[/yellow]")
            return 0
        chosen = manual
        console.print(f"[bold green]Selected:[/bold green] {chosen.slug}")

    art_dir = companion_art_dir(config_dir, slot)
    pending_dir = _pending_art_dir(art_dir)
    _discard_pending_art(pending_dir)

    try:
        paths = copy_premade(chosen, pending_dir)
    except KeyboardInterrupt:
        _discard_pending_art(pending_dir)
        console.print("[yellow]Copy cancelled. Existing art is unchanged.[/yellow]")
        return 130
    except (RuntimeError, OSError) as exc:
        _discard_pending_art(pending_dir)
        console.print(f"[red]Copy failed:[/red] {exc}")
        return 1
    except Exception:
        _discard_pending_art(pending_dir)
        raise

    _commit_pending_art(pending_dir, art_dir, config_dir, slot)
    console.print(f"[green]Copied {len(paths)} frames from premade {chosen.slug!r}.[/green]")
    console.print(
        "Run [bold]companion[/bold] to start the companion, or [bold]companion art[/bold] "
        "again to swap in real Gemini frames later."
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config_dir = _resolve_config_dir(args.config_dir)
    migrate_legacy_layout(config_dir)

    active_dir = get_active_companion_dir(config_dir)
    if active_dir is None:
        console.print("[red]No active companion.[/red] Run [bold]companion new[/bold] first.")
        return 1
    slot = active_dir.name
    companion = load_profile(active_dir / PROFILE_FILENAME)
    if companion is None:
        console.print(
            f"[red]Active companion '{slot}' has no readable profile.[/red] "
            "Run [bold]companion new[/bold] or [bold]companion roster[/bold] to fix it."
        )
        return 1

    _print_active_banner(companion)

    if _has_complete_art(companion_art_dir(config_dir, slot)):
        console.print(
            "[yellow]⚠[/yellow]  This companion already has a generated art set. "
            "Generating or prefilling will [bold]archive[/bold] it; you can restore via "
            "[bold]companion art → Restore[/bold] later."
        )

    has_archives = bool(_list_art_archives(config_dir, slot))
    has_sprite = _has_sprite_source(config_dir, slot)
    top = _pick_top_choice(allow_recrop=has_sprite, allow_restore=has_archives)
    if top is None or top == "quit":
        console.print("[yellow]Cancelled.[/yellow]")
        return 0

    if top == "premade":
        return _run_premade(config_dir, slot, companion)

    if top == "restore":
        return _run_restore(config_dir, slot)

    if top == "recrop":
        return _run_recrop(config_dir, slot)

    # top == "generate"
    mode = _pick_gen_mode()
    if mode is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return 0

    overrides: LocomotionOverrides | None = None
    if mode == "adapt":
        overrides = _collect_overrides(companion)
        if overrides is None:
            console.print("[yellow]Cancelled.[/yellow]")
            return 0

    api_key = _ensure_gemini_api_key(config_dir)
    if not api_key:
        console.print("[yellow]Aborted — no Gemini API key.[/yellow]")
        return 0

    return _run_generate(companion, config_dir, slot, overrides=overrides, api_key=api_key)
