"""``companion levelup`` — debug subcommand that levels the active companion.

Bypasses the eligibility check (``force=True``), runs the same
interactive picker that startup uses, saves the profile, and exits.
Does **not** start the Qt companion.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console

from claude_code_assist.cli._levelup_flow import run_levelup_interactive
from claude_code_assist.profile.storage import (
    PROFILE_FILENAME,
    get_active_companion_dir,
    load_profile,
    migrate_legacy_layout,
    save_profile,
)

logger = logging.getLogger(__name__)
console = Console()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="companion levelup",
        description=(
            "Force-level the active companion: pick a stat to boost, recompute "
            "the rarity, and reset the counters. Skips the eligibility check."
        ),
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
    profile_path = active_dir / PROFILE_FILENAME
    companion = load_profile(profile_path)
    if companion is None:
        console.print(
            f"[red]Active companion '{active_dir.name}' has no readable profile.[/red] "
            "Run [bold]companion roster[/bold] to fix it."
        )
        return 1

    if run_levelup_interactive(companion, force=True):
        save_profile(companion, profile_path)
        console.print(f"[green]Saved level-up to {profile_path}.[/green]")
    else:
        console.print("[yellow]No changes saved.[/yellow]")
    return 0
