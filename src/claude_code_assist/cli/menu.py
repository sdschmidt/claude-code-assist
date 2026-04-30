"""``companion`` (no args) — interactive top-level menu.

Shows a one-line banner for the active companion (or a hint to pick
one) and a select with start / new / roster / art / settings.

Loop semantics:

* ``new`` / ``roster`` / ``art`` / ``settings`` return to the menu so
  the user can chain actions without retyping ``companion``.
* ``start`` hands off to the Qt entry point and falls through to the
  shell when it exits — same as ``companion start`` from the command
  line.

Disable rules (each rendered with the reason next to the row):

* ``start`` — hidden art set (no ``frame_{0..9}.png``).
* ``art`` — no active companion.
* ``roster`` — empty roster.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import questionary
from rich.console import Console

from claude_code_assist.cli._picker import PICKER_STYLE, bind_shortcuts, menu_title
from claude_code_assist.profile.storage import (
    PROFILE_FILENAME,
    companion_art_dir,
    get_active_companion_dir,
    list_roster,
    load_profile,
    migrate_legacy_layout,
)

logger = logging.getLogger(__name__)
console = Console()

_YELLOW = "fg:ansiyellow bold"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="companion",
        description="Interactive companion launcher.",
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


def _has_complete_art(art_dir: Path) -> bool:
    if not art_dir.is_dir():
        return False
    return all((art_dir / f"frame_{i}.png").is_file() for i in range(10))


def _print_banner(config_dir: Path) -> None:
    """Render the top line: active companion, or the no-companion hint.

    Format matches ``companion roster`` / ``companion art``::

        Phlegmin  ★★  ·  Uncommon  ·  Lv. 3  ·  Gelatinous gremlin  ·  Sage
    """
    active_dir = get_active_companion_dir(config_dir)
    if active_dir is None:
        console.print(
            "[yellow]No companion selected.[/yellow] "
            "Use [bold]roster[/bold] to select one, or [bold]new[/bold] to create one."
        )
        return

    companion = load_profile(active_dir / PROFILE_FILENAME)
    if companion is None:
        console.print(
            f"[red]Active companion '{active_dir.name}' has no readable profile.[/red] "
            "Use [bold]roster[/bold] to pick another."
        )
        return

    from claude_code_assist.models.role import ROLE_CATALOG  # noqa: PLC0415
    from claude_code_assist.profile.leveling import format_xp_bar_segments, format_xp_percent  # noqa: PLC0415

    color = companion.rarity.color
    rarity_name = companion.rarity.value.title()
    role_block = ""
    if companion.role is not None:
        defn = ROLE_CATALOG.get(companion.role)
        role_color = defn.color if defn else "#888"
        role_block = f"  [dim]·[/dim]  [{role_color}]{companion.role.value}[/{role_color}]"
    console.print(
        f"[bold {color}]{companion.name}[/bold {color}]"
        f"  [{color}]{companion.rarity.stars}[/{color}]"
        f"  [{color}]·  {rarity_name}[/{color}]"
        f"  [dim]·  Lv. {companion.level}  ·  {companion.creature_type}[/dim]"
        f"{role_block}"
    )
    # XP bar mirrors the tray header: rarity-colored fill, dim background,
    # trailing percentage.
    xp_filled, xp_empty = format_xp_bar_segments(companion.comment_counter)
    xp_percent = format_xp_percent(companion.comment_counter)
    console.print(f"[dim]XP[/dim] [{color}]{xp_filled}[/{color}][dim]{xp_empty}[/dim] [dim]{xp_percent}[/dim]")


def _build_choices(
    config_dir: Path,
) -> tuple[list[questionary.Choice], str, dict[str, str]]:
    """Return (choices, default_value, shortcut_map).

    ``shortcut_map`` maps single-key shortcuts to the ``Choice.value`` they
    should resolve to; bound post-build via :func:`bind_shortcuts` so the
    user can press a letter to submit immediately.
    """
    active_dir = get_active_companion_dir(config_dir)
    has_companion = active_dir is not None
    has_art = has_companion and _has_complete_art(companion_art_dir(config_dir, active_dir.name))
    roster_count = len(list_roster(config_dir))

    start_disabled = None
    if not has_companion:
        start_disabled = "no companion selected"
    elif not has_art:
        start_disabled = "no art — run `art` first"

    art_disabled = None if has_companion else "no companion selected"
    roster_disabled = None if roster_count > 0 else "roster is empty"

    if not has_companion:
        default_action = "new"
    elif not has_art:
        default_action = "art"
    else:
        default_action = "start"

    # (value, shortcut, label, description, label_style or "" for default).
    # ``label_style`` only fires for enabled rows; disabled rows fall through
    # to a plain-string title so ``class:disabled`` colours the whole row.
    rows: list[tuple[str, str, str, str, str | None]] = [
        ("start", "s", "start", "launch the companion", _YELLOW if not start_disabled else None),
    ]
    if _is_levelup_eligible(config_dir):
        rows.append(("levelup", "l", "levelup", "apply the pending level-up", _YELLOW))
    rows.extend(
        [
            ("new", "n", "new", "create a new companion", _YELLOW if default_action == "new" else ""),
            ("roster", "r", "roster", "browse and switch companions", "" if not roster_disabled else None),
            ("art", "a", "art", "generate sprite frames", _YELLOW if default_action == "art" else ""),
            ("settings", "e", "settings", "edit gravity / scale / cooldown", ""),
            ("quit", "q", "quit", "exit", ""),
        ]
    )

    disabled_for: dict[str, str | None] = {
        "start": start_disabled,
        "roster": roster_disabled,
        "art": art_disabled,
    }

    choices: list[questionary.Choice] = []
    shortcut_map: dict[str, str] = {}
    for value, key, label, description, label_style in rows:
        reason = disabled_for.get(value)
        if reason:
            # Plain string lets questionary's ``class:disabled`` colour the
            # label + appended ``(reason)`` together. We drop the description.
            choices.append(questionary.Choice(title=label, value=value, disabled=reason))
            continue
        title = menu_title(label, description, shortcut=key, label_style=label_style or "")
        choices.append(questionary.Choice(title=title, value=value))
        shortcut_map[key] = value

    return choices, default_action, shortcut_map


def _is_levelup_eligible(config_dir: Path) -> bool:
    """Return True iff the active companion has a level-up waiting.

    Best-effort: returns False on a missing / unreadable profile so the
    menu never shows a leveldup row that would crash on dispatch.
    """
    active_dir = get_active_companion_dir(config_dir)
    if active_dir is None:
        return False
    companion = load_profile(active_dir / PROFILE_FILENAME)
    if companion is None:
        return False
    from claude_code_assist.profile.leveling import is_eligible_for_levelup  # noqa: PLC0415

    return is_eligible_for_levelup(companion)


def _print_warnings(config_dir: Path) -> None:
    """Surface the why-is-start-disabled hint above the picker."""
    active_dir = get_active_companion_dir(config_dir)
    if active_dir is None:
        return
    if not _has_complete_art(companion_art_dir(config_dir, active_dir.name)):
        console.print(
            "[yellow]⚠[/yellow]  No sprite frames yet — "
            "[bold]start[/bold] is disabled. Run [bold]art[/bold] to generate or prefill them."
        )


def _forward_args(config_dir: Path | None, debug: bool) -> list[str]:
    """Build the argv list to pass to subcommand ``run`` functions."""
    out: list[str] = []
    if config_dir is not None:
        out.extend(["--config-dir", str(config_dir)])
    if debug:
        out.append("--debug")
    return out


def _dispatch(action: str, config_dir: Path | None, debug: bool) -> int:
    forwarded = _forward_args(config_dir, debug)
    if action == "new":
        from claude_code_assist.cli.new import run as run_new

        return run_new(forwarded)
    if action == "roster":
        from claude_code_assist.cli.roster import run as run_roster

        return run_roster(forwarded)
    if action == "art":
        from claude_code_assist.cli.art import run as run_art

        return run_art(forwarded)
    if action == "settings":
        from claude_code_assist.cli.settings import run as run_settings

        return run_settings(forwarded)
    if action == "levelup":
        from claude_code_assist.cli.levelup import run as run_levelup

        return run_levelup(forwarded)
    return 0


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config_dir = _resolve_config_dir(args.config_dir)
    migrate_legacy_layout(config_dir)

    skip_clear = False
    while True:
        if skip_clear:
            # Last subcommand failed; leave its message at the top of the
            # scrollback and just re-render the menu below.
            console.print()
            skip_clear = False
        else:
            console.clear()
        _print_banner(config_dir)
        _print_warnings(config_dir)

        choices, default_action, shortcut_map = _build_choices(config_dir)
        try:
            question = questionary.select(
                "What now?",
                choices=choices,
                default=default_action,
                style=PICKER_STYLE,
            )
            bind_shortcuts(question, shortcut_map)
            choice = question.ask()
        except (KeyboardInterrupt, EOFError):
            choice = None

        if choice is None or choice == "quit":
            return 0

        console.clear()

        if choice == "start":
            from claude_code_assist.qt.app import main as run_companion

            forwarded: list[str] = []
            if args.config_dir is not None:
                forwarded.extend(["--config-dir", str(args.config_dir)])
            if args.debug:
                forwarded.append("--debug")
            return run_companion(forwarded)

        rc = _dispatch(choice, args.config_dir, args.debug)
        if rc != 0:
            # Skip the ``console.clear()`` next iteration so the failure
            # message (e.g. Gemini's 429) stays on screen above the menu.
            skip_clear = True
